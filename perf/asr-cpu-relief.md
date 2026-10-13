# ASR on the NPU: CPU relief and the offload thresholds

Offloading the whisper.cpp encoder to the NPU frees more CPU than it saves in wall-clock time.
On `base.en`, a 1.10x wall-clock speedup at 30 s is a 17.4% CPU saving [HW sweep]. A
streaming-ASR caller that shares the CPU gets back the core-seconds freed, and the speedup does
not measure them.

**Conditions:** RK3588 (Turing RK1), NPU clock 600 MHz, the vendor `rknpu` 0.9.8 driver through
`rknpu-submit`, 2026-08-24, unless a section states its own. The process runs 4 threads pinned to
the A76 cores (`taskset -c 4-7`), under CPU governor `performance`, with `ROCKET_KACC=1`. The
CPU-only configuration is the same binary with every op declined by the backend.

**Not characterized:** audio below 1.3 s, and a word error rate scored against references for
the CTC models. Nor are the two Q8_0 CTC models with `ROCKET_MIN_M` below 128, or the 1.3-2.5 s
utterances' cost at the default `ROCKET_MIN_M` of 32.

whisper.cpp, `base.en`, clean speech [HW sweep]:

| clip | wall-clock, CPU | wall-clock, NPU | speedup | core-s CPU | core-s NPU | CPU freed |
|---|---:|---:|---:|---:|---:|---:|
| 3 s | 2.13 | 1.68 | 1.27x | 7.56 | 4.40 | 41.8% |
| 30 s | 4.71 | 4.29 | 1.10x | 17.16 | 14.17 | 17.4% |
| 120 s | 14.78 | 12.54 | 1.18x | 55.78 | 42.51 | 23.8% |

## Cause

On this backend the wall-clock speedup understates the CPU saving. The backend moves the MACs to
the NPU and leaves the activation and weight packing (packA, packB) and the output de-tile on the
host. The NPU has no on-chip layout
conversion, so that host work is irreducible. Wall-clock time therefore includes time in which the
process is blocked on the NPU with its threads idle. The CPU that the offload frees is larger than
the wall-clock time it saves.

> [!NOTE]
> Depending on the shape, the wall-clock ratio overstates or understates the CPU saving. Measure
> CPU core-seconds where the CPU saving is the quantity of interest.

## Whisper relief by model size and speech density

The relief fraction rises with model size and falls with speech density. Whisper pads every
window to 30 s whatever the audio, so the encoder's cost is fixed per window and the decoder's is
not. Only the encoder offloads, so the encoder's share of the CPU core-seconds is the upper limit
on the relief. The encoder's share is largest for a big model on sparse audio.

CPU core-seconds freed, as clean speech / hard conversational speech [HW sweep]:

| model | 3 s | 10 s | 30 s | 60 s | 120 s |
|---|---:|---:|---:|---:|---:|
| tiny.en | 30.2 / 33.4% | 37.0 / 31.6% | 18.5 / 29.7% | 19.1 / 37.1% | 16.3 / 25.8% |
| base.en | 41.8 / 39.2% | 29.9 / 30.8% | 17.4 / 27.6% | 20.2 / 26.4% | 23.8 / 28.0% |
| small.en | 50.1 / 50.2% | 42.3 / 26.5% | 34.7 / 42.7% | 35.2 / 43.2% | 33.4 / 36.5% |

The NPU removes roughly half of the encoder's CPU cost, because the packing and the de-tile stay
on the host. It never removes more than the encoder's share.

The 3 s column is the largest relief for `base.en` and `small.en`, and the padding causes it. A
3 s clip incurs a full 30 s encode and decodes almost nothing, so the offloadable part is nearly
the whole cost. A short-utterance workload is at the favorable end of this axis.

The realtime factor is seconds of audio per second of wall-clock time. With the offload it is
3-33x for `tiny.en`, 1.8-12.9x for `base.en` and 0.7-5.8x for `small.en`. The sub-realtime test
points are the short clips, where the fixed 30 s encode and the model load dominate. The model
load is 129 ms for `tiny.en`, 173 ms for `base.en` and 333 ms for `small.en`, incurred once per
process.

## Offload thresholds

An op runs on the NPU only above a row-count threshold. The backend has two thresholds, and one
of them is set for a different workload:

- **`ROCKET_MIN_M` (default 32)** applies to F16/F32 weights. Whisper's models are F16 and its
  encoder runs 1500 rows, so the encoder always clears this threshold. The threshold keeps
  whisper's decoder on the CPU, which is its purpose. `whisper-cli` beam search batches 5
  decoders into one call, so every decode step is M=5. A threshold of 4 sends the whole decoder
  to the NPU at 2.3x slower than the CPU. The threshold also decides whether a CTC or transducer
  encoder's short utterance offloads, at 16-75 rows for 1-6 s of audio.
- **`ROCKET_MIN_M_QUANT` (default 512)** applies to quantized weights, which take a
  dequantize-to-fp16 path that needs more rows to amortize. Its default is set from LLM prefill,
  and it is too high for a CTC ASR encoder. A CTC encoder's sequence length is the audio length
  in frames: a few hundred for a clip of a few tens of seconds.

The threshold's effect is a step. The table gives CPU core-seconds freed against the CPU-only
configuration, for Q8_0 CTC models on conversational audio [HW sweep]:

| model | clip | `MIN_M_QUANT`=512 (default) | `MIN_M_QUANT`=128 |
|---|---|---:|---:|
| SenseVoice-small 234M | 3 s | -3.0% | -0.6% |
| | 10 s | 0.4% | **20.3%** |
| | 30 s | -0.9% | **42.3%** |
| | 60 s | 43.2% | 43.4% |
| | 120 s | 37.5% | 37.3% |
| Parakeet-CTC 0.6B | 3 s | -0.5% | 0.2% |
| | 10 s | -0.4% | 3.3% |
| | 30 s | 6.1% | **45.7%** |
| | 60 s | 46.5% | 46.7% |
| | 120 s | 47.3% | 47.4% |

**Conditions:** `ROCKET_MIN_M` at 128.

At 30 s under the default threshold, SenseVoice offloads no op, and `ROCKET_MM_PROFILE` prints
nothing. At 60 s it prints 1395 job-batches. Lowering the threshold costs nothing at any test
point where it does not help (worst -0.6%). From 60 s up the two configurations are equal, because
the sequence is over 512 either way.

Two effects keep these models' short clips on the CPU. Down to about 10 s, the quantized threshold
accounts for everything. Below that length the sequence is also under the F16 threshold of 128,
and `ROCKET_MIN_M_QUANT` cannot go lower than `ROCKET_MIN_M` (§"The quantized threshold's clamp"). At
3 s both configurations are within 3%.

The boundary below 10 s comes from the threshold rather than from the NPU. With `ROCKET_MIN_M`
itself lowered, an F16 Parakeet offloads and frees CPU down to 1.3 s
([the short-utterance measurements](#short-utterances-below-the-f16-floor)). For these two
Q8_0 models that outcome is unmeasured [expected], because they incur a per-call dequantization
that the F16 model does not.

### Threshold form

The offload threshold is a row count, and the weight size cancels out of it. The dequantization
that a forward pass incurs is fixed in the row count (§"The weight cache on a Q8_0 host")
[HW sweep].

Offloading one GEMM is a net gain under the condition `M*K*N*Δ > K*N*d + F`. Here `d` is the per-element
decode cost, `Δ` is the per-MAC advantage over the CPU, and `F` is the fixed per-call dispatch
cost. Dividing by `K*N` gives `M > d/Δ + F/(Δ*K*N)`. The weight size divides out of the decode
term, which the threshold exists to amortize. What remains is a constant in `M`, plus a correction
that shrinks as the weight grows.

A threshold of the form `K*N*sizeof(weight)/M` would therefore scale the wrong way. The
shape-aware refinement available is to let a large `K*N` clear a slightly lower threshold, not a
higher one.

The advantage `Δ` itself also falls at small `M`, where the NPU underuses its array, so in
principle the constant has a per-model component. With `ROCKET_MIN_M` at 128, the quantized
threshold cannot reach that component (§"The quantized threshold's clamp").

### The quantized threshold's clamp

**Conditions:** `ROCKET_MIN_M` at 128.

The function `rocket_min_m_quant()` never returns less than `rocket_min_m()`
[source-confirmed: `ggml-rocket.cpp`]. The `ROCKET_MIN_M_QUANT` values of 8, 32, 64, 96 and 128
are therefore the same value. The admitted-GEMM count is identical at all of them, for every model
and clip length tested. It is 279 for SenseVoice and 24 or 217 for Parakeet, never a count between
[HW sweep].

> [!NOTE]
> A `ROCKET_MIN_M_QUANT` sweep below `ROCKET_MIN_M` tests one point.

The threshold that sets the boundary is therefore the F16 threshold, and one constant places every
transition. The rule `M >= 128` predicts all three transitions, on two models whose frame rates
differ:

| family | rows | crosses 128 at | admitted below / above |
|---|---|---:|---|
| SenseVoice, all 279 GEMMs | `M = 16.6 s` | 7.8 s | 0 at 7 s, **279 at 8 s** |
| Parakeet, 193 of 217 | `M = 12.5 s` | 10.3 s | 24 at 10 s, **217 at 11 s** |
| Parakeet, the other 24 | `M = 2*12.5 s - 1` | 5.2 s | 0 at 5 s, 24 at 7 s |

The 24-GEMM family is one GEMM per block, with roughly twice the rows of the rest. Its shape
matches a relative-position projection over `2T-1` positions [hypothesis]. The row count is
measured, and the attribution is not.

### The crossover

**Conditions:** vendor RK1, A76-pinned, CPU governor `performance` at 2352 MHz, conversational
audio (the `aunt` clip set), `ROCKET_MIN_M` at 128.

The crossover is a step, with no clip length at which the offload is admitted and loses. The table
gives CPU core-seconds freed with the offload admitted, against the offload declined [HW sweep]:

| clip | 3 s | 5 s | 7 s | 10 s | 15 s | 20 s | 30 s | 45 s | 60 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SenseVoice-small 234M | 3.1% | -0.7% | 1.3% | **21.9%** | 31.0% | 34.4% | 42.5% | 45.0% | 43.1% |
| Parakeet-CTC 0.6B | 0.8% | 0.6% | 1.4% | 3.6% | **32.1%** | 40.0% | 43.4% | 46.0% | 46.8% |

The largest spread within one test point across the sweep is 11.9%. A test point under about 12%
therefore shows no measurable difference.

Where nothing is admitted, the two configurations are indistinguishable. The first test point that
admits anything frees 22-32%.

For this workload the threshold is a switch rather than a trade between a small gain and a small
loss at the margin. Neither model reaches a losing regime for it to exclude. Any setting at or
below 128 is the same setting, and the relief saturates at 43-47%.

Wall-clock time improves much less. At 30 s the speedup is 1.28x for SenseVoice and 1.33x for
Parakeet, against 42-43% of the CPU freed (§"Cause").

A threshold of 384 at 30 s admits Parakeet's 24-GEMM family and declines the other 193, which
separates their contributions. The 24 free 4.2% of the 43.4% that the full set frees, so 193 of
the 217 GEMMs provide roughly nine tenths of the benefit. The family that clears the threshold
earliest provides the smaller share.

## Short utterances below the F16 floor

**Conditions:** RK3588 (Turing RK1), mainline `rocket` 1.3.0, kernel 7.2.8, NPU clock 600 MHz,
2026-09-30. Parakeet TDT 0.6B v3 F16 runs through whisper.cpp v1.9.4's `parakeet-cli`, on 4
threads pinned to the A76 cores, under CPU governor `ondemand`. Each length is ten LibriSpeech
test-clean utterances. The NPU configuration runs `ROCKET_MIN_M=4`. Each figure is a
per-utterance cost with no model load in it.

A voice command is one to five seconds of audio. At 12.5 encoder frames a second, that is 16-75
rows. At the default `ROCKET_MIN_M` of 32, every encoder GEMM offloads from 2.6 s up, and 24 of
the 265 offload at 1.3-2.5 s (§"Setting `ROCKET_MIN_M`"). With the threshold lowered, every encoder GEMM runs on
the NPU and re-packs its weight on each call. The offload loses at no length measured
[HW sweep]. It ties the CPU-only wall-clock time at 1.3-1.8 s while freeing a third of the CPU,
and from 2 s up it is also faster.

| audio | rows | CPU wall-clock | NPU wall-clock | speedup | CPU core-s | NPU core-s | CPU freed |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.3-1.8 s | 16-23 | 516 ms | 520 ms | 1.00x | 2.01 | 1.32 | 34% |
| 2.1-2.5 s | 27-31 | 667 ms | 564 ms | 1.18x | 2.61 | 1.48 | 44% |
| 2.6-3.9 s | 33-49 | 938 ms | 673 ms | 1.41x | 3.71 | 1.73 | 54% |
| 4.4-5.9 s | 55-74 | 1449 ms | 838 ms | 1.73x | 5.73 | 2.21 | 61% |
| 7.1-8.9 s | 88-112 | 2278 ms | 1139 ms | 2.01x | 9.03 | 3.03 | 66% |
| 11.4-13.9 s | 143-174 | 3724 ms | 1659 ms | 2.25x | 14.77 | 4.51 | 69% |

The row column is the audio at 12.5 frames a second. At a threshold of 128, the NPU
configuration measures within 1% of the CPU-only one through 5.9 s. It admits no GEMM through
3.9 s. At 4.4-5.9 s only the longer of the two utterances admits the 24 GEMMs of the `2T-1`
family [host-computed: the admission profile]. It frees 4% at 7-9 s, where both utterances clear
128 rows, and matches the lowered-threshold configuration from 11 s.

Every configuration's transcript is
byte-identical to the CPU-only one at every length. The word error rate against the references is 13.6, 1.9, 2.5,
0.7, 0.9 and 0.5% by length, and the shortest set's errors are proper names.

### The OpenMP wait policy

`OMP_WAIT_POLICY=PASSIVE` takes a further 7-17% off the NPU configuration, which then frees
45-72%. The NPU configuration's threads wait on the NPU. On the CPU-only configuration the same
setting leaves the core-seconds within 1% at every length, and costs it 2-5% of wall-clock time
[HW sweep].

### A second process on the same cores

**Conditions:** a continuous stream of 2.1-2.5 s utterances runs in one `parakeet-cli`
ASR process.
Beside it runs `llama-bench` decode of Llama-3.2-3B `Q4_K_M`, on the CPU at 4 threads. Both are
pinned to the same four A76 cores. The ASR columns give the median utterance over the stream, and
its ratio to the uncontended cost above.

Freed core-seconds help only a process that can use them. The first table measures the ASR stream
and the LLM decode side by side [HW sweep 2026-09-30]:

| ASR stream | LLM decode | against alone | ASR per utterance | against uncontended |
|---|---:|---:|---:|---:|
| none | 7.07 t/s | 1.00x | n/a | n/a |
| CPU | 3.26 t/s | 0.46x | 1.19 s | 1.8x |
| NPU, threshold 4 | 4.89 t/s | 0.69x | 2.39 s | 4.2x |
| NPU, threshold 4, passive | 5.62 t/s | 0.79x | 1.98 s | 3.7x |

The offload gives the other process 1.5-1.7x the throughput, at the cost of the offloaded stream's
own latency. Under contention the CPU stream slows 1.8x and the offloaded one 3.7-4.2x, so the
offloaded utterance becomes the slower of the two. Each of an utterance's 265 offloaded GEMMs
packs its weight and submits from a host thread. Each of those steps waits for a core while the
NPU idles [hypothesis].

A voice assistant that protects its listener lowers the priority of other work. The second table
puts the LLM at the lowest priority [HW sweep 2026-09-30]:

| ASR stream | LLM decode | against alone | ASR per utterance | against uncontended |
|---|---:|---:|---:|---:|
| none | 7.23 t/s | 1.00x | n/a | n/a |
| CPU | 0.90 t/s | 0.12x | 0.78 s | 1.17x |
| NPU, threshold 4 | 3.37 t/s | 0.47x | 1.54 s | 2.7x |
| NPU, threshold 4, passive | 4.80 t/s | 0.66x | 0.89 s | 1.66x |

**Conditions:** as above, with the LLM at `nice -n 19` and running `-n 16 -r 2` beside an
800-utterance stream, which outlived it in every configuration.

Priority protects the CPU stream and starves the LLM to an eighth, but it does not protect the
offloaded stream at the default wait policy. Under `OMP_WAIT_POLICY=PASSIVE`, the offloaded stream
costs the LLM a third of its rate, and the CPU stream costs it seven eighths. The cost to the ASR
stream is a 14% slower utterance, 0.89 s against 0.78. The likely cause is that the ASR process's
OpenMP threads spin while its NPU workers wait for a core [hypothesis]. The passive policy puts
them to sleep.

### Where the floor comes from

The default `ROCKET_MIN_M` of 32 is above the F16 crossover measured on LLM prefill
[HW sweep 2026-10-09]:

| prompt tokens | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B F16 | 0.75x | 1.20x | 1.71x | 1.96x |
| Llama-3.2-3B F16 | 0.78x | 1.40x | 2.51x | 3.68x |
| Ministral-3-8B F16 | 0.77x | 1.46x | 2.75x | 3.90x |

**Conditions:** `llama-bench` on llama.cpp a7fb71f, the NPU at a threshold of 4 against the faster
of 4 and 8 CPU threads, CPU governor `performance`.

The LLMs cross at 21-25 rows, and Parakeet below 16. On a 176 s clip, whisper-small F16 at 32 runs
0.995-1.006x its wall-clock time at 128, with the transcript byte-identical. Its decoder's prompt
passes of 64-127 rows offload at 32 and tie. The narrower models tie as well. Under `ondemand`,
tiny runs 1.000-1.007x and base 1.005-1.014x at 32 against 128 [HW sweep 2026-10-09].

### The per-call weight pack

Below 256 rows no weight is made resident. Of the 265 GEMMs an utterance runs, 120 also have no
stable weight name. At these lengths every weight is therefore re-packed on every call. The
`ROCKET_MM_PROFILE` readout puts `packB` at 0.59-0.60 thread-seconds an utterance at 1.3-2.5 s,
against the NPU configuration's 1.32-1.48 core-seconds [HW sweep]. A profile bucket is a thread
interval. The `packB` figure is therefore an upper bound on what a resident path for small row
counts could remove, not an estimate of it.

### An int8 CPU runtime

The runtime sherpa-onnx 1.13.8 runs Parakeet TDT 0.6B v3 as an int8 ONNX export on ONNX Runtime.
It is a faster CPU path for this model than the F16 CPU-only configuration [HW sweep 2026-09-30]:

| audio | sherpa-onnx int8, CPU | F16, CPU | F16, NPU threshold 4, passive |
|---|---:|---:|---:|
| 1.3-1.8 s | 231 ms, 0.92 core-s | 516 ms, 2.01 | 489 ms, 1.09 |
| 2.1-2.5 s | 290 ms, 1.15 | 667 ms, 2.61 | 537 ms, 1.24 |
| 2.6-3.9 s | 400 ms, 1.61 | 938 ms, 3.71 | 640 ms, 1.46 |
| 4.4-5.9 s | 768 ms, 2.69 | 1449 ms, 5.73 | 811 ms, 1.93 |
| 11.4-13.9 s | 1261 ms, 5.01 | 3724 ms, 14.77 | 1681 ms, 4.20 |

**Conditions:** one persistent process on the same four A76 threads and the same utterances.

On a voice command the int8 runtime is about twice as fast as the NPU configuration. It uses 7-16%
less CPU below 2.5 s, and 10-39% more from 2.6 s up. Its per-utterance wall-clock times are bimodal
under `ondemand`: the same 4.4-5.9 s utterance measured 0.53 s in one pass and 1.02 s in another.
A pinned governor is expected to hold its fast mode [expected]. Its word error rate on the
1.3-1.8 s set is 27.3% against the F16 model's 13.6%. Over 22 words, the difference is six errors
against three.

### Setting `ROCKET_MIN_M`

`ROCKET_MIN_M=16` admits the same 265 GEMMs as a threshold of 4 at every length measured, down to
1.3 s [HW sweep]. A threshold of 32 admits 24 of them on the 1.3-2.5 s utterances. A threshold of
16 still keeps whisper.cpp's beam-5 decode, 5 rows a step, on the CPU. Set it in the ASR process
only, because the 3B loses at 16 rows (0.78x in the LLM table above).

## Thread count

The thread count changes wall-clock time and leaves the CPU core-seconds unchanged, because the
host work is the same whatever number of threads performs it. The thread count therefore trades
realtime headroom against latency, and the CPU saving stays. The table is `base.en`, ten 3 s
utterances, with the offload [HW sweep]:

| threads | CPU core-seconds | realtime | CPU-only core-seconds |
|---|---:|---:|---:|
| 1 | 41.96 | 0.76x | 66.25 |
| 2 | 39.03 | 1.42x | 67.60 |
| 4 | 40.39 | 2.36x | 67.67 |

The core-seconds are flat within 7% with the offload and within 2% without it, while wall-clock
time changes 3.1x. The CPU saving holds at 37-42% at every thread count. The realtime factor does
change: at one thread `base.en` falls below realtime.

> [!NOTE]
> Reducing the thread count to protect other work on the board keeps the CPU saving, and can drop
> the transcription below realtime.

## Numerical faithfulness

The NPU encoder runs in fp16 where the CPU's runs in fp32, so its output is close to the CPU's
and not bit-identical. Whether the difference changes the transcript depends on the clip length
and on how well-conditioned the audio is. Both configurations are deterministic (every repetition
within one is byte-identical), so a difference between them comes from the numerics, not from a
race.

Word-level divergence between the two configurations, whisper.cpp [HW sweep]:

| audio | tiny.en | base.en | small.en |
|---|---|---|---|
| clean read speech, <= 60 s | identical | identical | identical |
| clean read speech, 120 s | diverges in a repetition loop | identical | identical |
| hard conversational, <= 10 s | identical | identical | identical |
| hard conversational, 30-60 s | identical | 4-6 word edits | identical |
| hard conversational, 120 s | identical | 66 edits of 169 words | 40 edits of 100 |

Neither configuration is ground truth. On the hard clip both produce plausible readings of
ambiguous overlapping speech, and they differ in both directions. A perturbation early in an
autoregressive decode changes the prompt context for every window after it. The divergence
therefore compounds with length rather than staying local.

> [!NOTE]
> A transcript test on clean audio does not certify degraded audio, which is the case of a radio
> or telephony caller.

On a CTC encoder the divergence does not compound with length, because a CTC decode has no
context to propagate it. The two Q8_0 CTC models were compared offloaded against declined, on hard
conversational audio of 3-60 s [HW sweep]. Every test point that offloads nothing is
byte-identical. Every test point that offloads differs by one or two words in roughly
twenty-five. SenseVoice at 10 s transcribes `regaul` against `reaginaul`, and at 30 s `too`
against `two`. Parakeet at 30 s inserts a spurious `k`.

No reference transcript was scored, so the comparison bounds the size of the disagreement and does
not say which configuration is better. The disagreement has a different shape from whisper's.
Parakeet is byte-identical at 60 s while differing at 30 s and 45 s, so the edits stay local and
their count does not grow with the clip. Across the whole sweep, no test point produced more than
one distinct transcript over its repetitions.

## Per-utterance cost with the model already loaded

A streaming caller loads the model once. The single-shot figures in the earlier sections include
the model load, which a server incurs once rather than per utterance. The table gives steady-state
CPU core-seconds per utterance on conversational audio. Each entry is the CPU-only cost, then the
cost with the offload, then the share freed [HW sweep]:

| utterance | whisper tiny.en | whisper base.en | SenseVoice-small |
|---|---|---|---|
| 3 s | 3.03 -> 1.93 (36%) | 6.75 -> 3.82 (43%) | 1.34 -> 1.34 (0%) |
| 10 s | 3.39 -> 2.45 (28%) | 8.37 -> 4.89 (42%) | 4.67 -> 3.27 (30%) |
| 30 s | 4.38 -> 3.11 (29%) | 9.44 -> 6.17 (35%) | 13.37 -> 7.42 (44%) |

The two model families cross. A CTC model's cost follows the audio length, so it is much the
cheapest on a short utterance. At the default threshold, a short utterance is exactly where the
NPU does not reach it.

Whisper incurs a fixed 30 s encode whatever the utterance. The fixed encode wastes CPU on a 3 s
clip, and it gives the NPU a job large enough to gain every time. By 30 s, whisper `base.en` with
the offload costs less absolute CPU than SenseVoice does. Whisper's cost is flat in utterance
length, and the CTC model's is linear.

## The weight cache on an F16 host

On an F16 host a weight cache removes little, because the per-call weight cost is small. The
function `rocket_weight_key()` rejects ggml's positional autonames (`leaf_%d`). Those names come
from whisper.cpp, which never names its weight tensors, so every weight is re-packed on every call
rather than cached. The function `mm_pack_weights()` covers the prep, the scatter and the fini, so
`packB` is the whole per-call weight cost [HW sweep]:

| model | clip | packB | whole-process core-s | packB share |
|---|---|---:|---:|---:|
| tiny.en | 120 s | 43 ms | 19.8 | 0.22% |
| base.en | 120 s | 118 ms | 42.7 | 0.28% |
| small.en | 120 s | 551 ms | 97.0 | 0.57% |

The pack is at most 0.6% of the process CPU and at most ~4.5% of the encode wall-clock time. A
cache keyed on the weights buffer, measured on the mainline board, keys correctly, keeps the
transcripts md5-identical and takes packB to zero. The total is 1.3% worse, because the resident
path's extra BO sync costs more than the pack it removes.

The F16 weight-residency gain on LLM prefill (+18.5% at pp512, +13.6% at pp2048) does not transfer
to this workload. An LLM's weights are enormous against one micro-batch of activations, while a
whisper encoder's weights are small against 1500 rows of activations.

The dominant host cost is the per-worker input load. For `base.en` at 120 s, the pack is 1244 ms,
of which 103 ms is the shared A scatter and 118 ms is the weight scatter. The rest is the
per-worker copy of the input cube.

The pack share caps a cache's gain only on F16 weights, where `weight_dequant` is 0 by
construction. A Q8_0 host incurs `weight_dequant` on every call, and a resident cache would remove the dequantization and the scatter
together. The next section measures that term, whose share differs.

## The weight cache on a Q8_0 host

On a Q8_0 host a weight cache removes `weight_dequant` + `packB`, a term fixed per forward pass.
A cache saves it only across utterances. On short clips its share is an order of magnitude larger
than the F16 cap.

The table covers two Q8_0 CTC encoders through `transcribe.cpp` [HW sweep 2026-08-26]:

| model | clip | CPU-only core-s | NPU core-s | relief | dequant | packB | GEMM calls | cache cap (of NPU core-s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SenseVoiceSmall-Q8_0 | 10 s | 4.57 | 3.54 | 22.6% | 293 ms | 138 ms | 279 | **12.2%** |
| | 30 s | 13.59 | 8.01 | 41.1% | 264 ms | 152 ms | 279 | 5.2% |
| | 60 s | 32.12 | 18.41 | 42.7% | 246 ms | 173 ms | 279 | 2.3% |
| | 120 s | 86.74 | 54.28 | 37.4% | 249 ms | 160 ms | 279 | 0.8% |
| parakeet-ctc-0.6b-Q8_0 | 10 s | 8.97 | 8.41 | 6.2% | 29 ms | 27 ms | 24 | 0.7% |
| | 30 s | 25.62 | 14.36 | 43.9% | 584 ms | 538 ms | 217 | **7.8%** |
| | 60 s | 56.09 | 30.11 | 46.3% | 580 ms | 527 ms | 217 | 3.7% |
| | 120 s | 136.07 | 72.11 | 47.0% | 578 ms | 582 ms | 217 | 1.6% |

**Conditions:** vendor RK1, A76-pinned, CPU governor `performance`, the NPU configuration at
`ROCKET_MIN_M_QUANT=128` (the threshold at which this workload offloads).

The term is fixed per forward pass, as the GEMM-call column shows. SenseVoice offloads 279 GEMMs
whether the clip is 10 s or 120 s, and its dequantization stays within 246-293 ms across a 12x
range of audio. Parakeet is the same from 30 s up, at 217. The share therefore falls with clip
length, from 12.2% to 0.8%, and the cache is an optimization for short utterances.

Within one utterance there is nothing to cache. The 279 calls are 279 distinct GEMMs, and each
dequantizes its own weight once. The count does not grow with audio length, so a single-clip run
has no reuse to exploit. The saving exists only across utterances, in a process that keeps the
model loaded. A streaming service runs in that steady state, which §"Per-utterance cost with the
model already loaded" measures. A one-shot CLI invocation would gain nothing.

The cap is a property of the model, not of the model family. Parakeet at 10 s offloads 24 GEMMs
rather than 217 and frees 6.2%, because most of its layers are under the threshold at that
length. Its cache cap there is 0.7%.

> [!NOTE]
> Measure the cache cap on the candidate model at the candidate utterance length.

## NPU clock parking on the vendor driver

On the vendor driver the NPU clock does not drop to its idle rate during an encode. Mainline
`rocket` defaults to a 50 ms `autosuspend_delay_ms`, and host packing leaves gaps longer than
that. On mainline, all three cores therefore idle inside a single encode, and the next submit runs
at a third of the clock. The patch `patches/rocket/081` raises the delay to 1000 ms.

The vendor `rknpu` driver sets `power_put_delay = 3000`, and its `rknpu_ondemand` governor never
measures load, so neither effect exists on the vendor driver. Through one `small.en` encode the
clock holds one frequency [HW sweep].

## Evidence

The raw sweeps, the harnesses and the clip generator are in [data/asr/](data/asr/).

### Measurement method

Both configurations of every comparison come from one binary. For whisper.cpp the CPU-only
configuration is `ROCKET_MIN_M=1000000 ROCKET_FLASH_ATTN=0`, under which every op declines at
`supports_op`. For transcribe.cpp it is `--backend cpu` against `--backend cpu_accel`. The warm-up
run is discarded, and the configurations are interleaved with the order alternating [HW sweep].

### Offload thresholds

The step table's three configurations come from one binary.

### The quantized threshold's clamp

The 8 s and 11 s test points tested the constant. They ran after it was fitted on the other test
points, and both returned the predicted count.

### The crossover

Both configurations run through the same backend, so admission is the only variable. Each test
point is the median of 3 warm repetitions, with the first process and the first run of each
discarded. The CPU frequency was read back from `scaling_cur_freq`.

### Short utterances below the F16 floor

An utterance's cost is a 10-file process minus a 1-file process, over 9. Three passes rotate the
configuration order, and the table is their medians, every pass within 6%. The transcripts were
compared over all three passes.

### A second process on the same cores

The stream starts 4 s before the neighbor. The first table is three passes with the configuration
order rotated, and the second table two passes. The second table's two NPU configurations had short
windows, 8-10 utterances each, so their ASR columns are noisier.

### Where the floor comes from

The two LLM passes agreed within 0.04. The whisper-small comparison ran three passes. On this
harness the same work at 32 and 64 spreads 0.998-1.011x, and every transcript is byte-identical.

### An int8 CPU runtime

The comparison ran three passes, with the first dropped.

### Per-utterance cost with the model already loaded

The harnesses are `whisper-cli` with the clip repeated across `-f` arguments, and
`transcribe-cli --batch`, which reuses one context.

### The weight cache on a Q8_0 host

Each test point is the median of two interleaved repetitions, with a warm-up discarded.

### NPU clock parking on the vendor driver

The clock was sampled every 50 ms through the encode: 231 samples, one distinct frequency, zero
transitions.

### Revision history

The F16 prefill crossover on an earlier measurement:

| Measured | Quantity | Value | Conditions |
|---|---|---|---|
| 2026-09-30 | Llama-3.2-3B F16 prefill, the NPU at a threshold of 4 against the CPU | 0.70x at 16 rows, 1.25x at 32, 2.34x at 64, 3.58x at 128 | llama.cpp b11242, the CPU at 4 threads, two passes within 2% [HW sweep 2026-09-30] |

The 2026-10-09 table in §"Where the floor comes from" measures against the faster of 4 and 8 CPU
threads, and gives 0.78x at 16 rows.

### Excluded causes

- **Spin in the CPU-only configuration.** Its user time is four threads times its wall-clock time,
  which looks like OpenMP spin. The passive wait policy does not reduce its core-seconds
  (§"The OpenMP wait policy"), so that time is work, and the relief is not the CPU-only
  configuration's spin.
