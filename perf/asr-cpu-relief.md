# ASR on the NPU: CPU relief and the offload floors

Speech-to-text is the first workload these notes measure in CPU core-seconds rather than
in throughput. The question a streaming-ASR caller asks is not "how fast" but "how much of
the CPU do I get back". The two are different numbers, and the difference is not small.

The measurements ran on an RK3588 (Turing RK1) at 600 MHz, on the vendor `rknpu` 0.9.8
driver through `rknpu-submit`. The process was A76-pinned (`taskset -c 4-7`) with 4
threads, `ROCKET_KACC=1`, and the CPU governor `performance`. Both arms of every A/B come
from one binary. For whisper.cpp the CPU arm is `ROCKET_MIN_M=1000000 ROCKET_FLASH_ATTN=0`,
under which every op declines at `supports_op`. For transcribe.cpp it is `--backend cpu`
against `--backend cpu_accel`. The warm-up run is discarded, and the arms are interleaved
with the order alternating [HW sweep, 2026-08-24].

The raw sweeps, the harnesses and the clip generator are in [data/asr/](data/asr/).

## Wall speedup and CPU saving

The wall speedup is not the CPU saving, and on this backend it understates the saving. This
backend moves the MACs to the NPU and leaves packA, packB and the output de-tile on the
host. The NPU has no on-chip layout conversion, so that host work is irreducible. The wall
therefore contains time that the process spends blocked on the NPU with its threads idle.
So the CPU that the offload frees is larger than the wall it saves.

whisper.cpp, `base.en`, clean speech [HW sweep]:

| clip | wall CPU | wall NPU | wall x | core-s CPU | core-s NPU | CPU freed |
|---|---:|---:|---:|---:|---:|---:|
| 3 s | 2.13 | 1.68 | 1.27x | 7.56 | 4.40 | 41.8% |
| 30 s | 4.71 | 4.29 | 1.10x | 17.16 | 14.17 | 17.4% |
| 120 s | 14.78 | 12.54 | 1.18x | 55.78 | 42.51 | 23.8% |

A 1.10x wall at 30 s is a 17.4% CPU saving. Quoting the wall ratio as the CPU saving is
wrong in both directions, depending on the shape. Measure the quantity that the caller
asks about.

## Whisper relief by model size and speech density

The relief fraction rises with model size and falls with speech density. Whisper pads every
window to 30 s whatever the audio, so the encoder's cost is fixed per window and the
decoder's is not. The encoder is the only part that offloads, so the encoder's share of
the CPU core-seconds is the ceiling on the relief. That share is largest for a big model
on sparse audio.

CPU core-seconds freed, as clean speech / hard conversational speech [HW sweep]:

| model | 3 s | 10 s | 30 s | 60 s | 120 s |
|---|---:|---:|---:|---:|---:|
| tiny.en | 30.2 / 33.4% | 37.0 / 31.6% | 18.5 / 29.7% | 19.1 / 37.1% | 16.3 / 25.8% |
| base.en | 41.8 / 39.2% | 29.9 / 30.8% | 17.4 / 27.6% | 20.2 / 26.4% | 23.8 / 28.0% |
| small.en | 50.1 / 50.2% | 42.3 / 26.5% | 34.7 / 42.7% | 35.2 / 43.2% | 33.4 / 36.5% |

The NPU removes roughly half of the encoder's CPU cost, not all of it, because the packing
and the de-tile stay on the host. It never removes more than the encoder's share.

The 3 s column is the largest relief for every model, and the padding causes it. A 3 s
clip pays a full 30 s encode and decodes almost nothing, so the offloadable part is nearly
the whole cost. A short-utterance workload sits at the favorable end of this axis, not the
unfavorable one.

The realtime factor is seconds of audio per second of wall. On the NPU arm it is 3-33x for
`tiny.en`, 1.8-12.9x for `base.en` and 0.7-5.8x for `small.en`. The sub-realtime cells are
the short clips, where the fixed 30 s encode and the model load dominate. The model load is
129 ms for `tiny.en`, 173 ms for `base.en` and 333 ms for `small.en`, paid per process.

## The offload floors

An op reaches the NPU only above a row floor. There are two floors, and one of them is set
for a different workload:

- **`ROCKET_MIN_M` (default 128)** applies to F16/F32 weights. Whisper's models are F16 and
  its encoder runs 1500 rows, so this floor never binds on the encoder. It binds on
  whisper's decoder, which is its purpose. `whisper-cli` beam search batches 5 decoders
  into one call, so every decode step is M=5. A floor of 4 sends the whole decoder to the
  NPU at 2.3x slower than the CPU. The floor also binds on a CTC or transducer encoder's
  short utterance, which runs 16-75 rows at 1-6 s of audio.
- **`ROCKET_MIN_M_QUANT` (default 512)** applies to quantized weights, which take a
  dequantize-to-fp16 path that needs more rows to amortize. **This floor is set from LLM
  prefill, and it is the wrong floor for a CTC ASR encoder.** A CTC encoder's sequence
  length is the audio length in frames: a few hundred for a clip of a few tens of seconds.

The consequence is a step, not a ramp. The table gives CPU core-seconds freed against the
CPU arm, for Q8_0 CTC models on conversational audio, with three arms from one binary
[HW sweep]:

| model | clip | `MIN_M_QUANT`=512 (shipped) | `MIN_M_QUANT`=128 |
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

At 30 s, `ROCKET_MM_PROFILE` prints nothing at all for SenseVoice under the shipped floor.
That is not a small win, it is zero offloaded ops. At 60 s it prints 1395 job-batches.
Lowering the floor costs nothing in any cell where it does not help (worst cell -0.6%).
Above 60 s the two arms are equal, because the sequence is over 512 either way.

So the reading "NPU buys nothing on short audio for these models" is two effects, not one.
The quantized knob accounts for everything down to about 10 s. Below that length the
sequence also sits under the F16 floor, and `ROCKET_MIN_M_QUANT` cannot go lower than that
floor (§"The quantized floor's clamp"). At 3 s both arms are within 3%.

That boundary is the floor's and not the NPU's. With `ROCKET_MIN_M` itself lowered, an F16
Parakeet offloads and wins down to 1.3 s (see
[Short utterances below the F16 floor](#short-utterances-below-the-f16-floor)). For these
two Q8_0 models that outcome is unmeasured [expected], because they pay a per-call dequant
that the F16 model does not.

### The floor's dependence on weight size

A row floor is the right shape for the gate, and the weight size cancels out of it. The
tempting alternative is a floor expressed against the weight it guards. The argument is
that a quantized weight costs more to decode, so a bigger weight needs more rows. That
argument is wrong, and the profile shows why. The dequant that a pass pays is fixed in the
row count. SenseVoice offloads 279 GEMMs and spends 250-293 ms decoding their weights
whether the clip is 10 s or 120 s [HW sweep].

Offloading one GEMM wins under the criterion `M*K*N*Δ > K*N*d + F`. Here `d` is the
per-element decode cost, `Δ` is the per-MAC advantage over the CPU, and `F` is the fixed
per-call dispatch. Dividing by `K*N` gives `M > d/Δ + F/(Δ*K*N)`. The weight size divides
out of the term that the floor exists to amortize. What remains is a constant in `M` plus
a correction that shrinks as the weight grows.

So a floor of the form `K*N*sizeof(weight)/M` would scale the threshold the wrong way. A
row floor is the right primary shape. The available shape-aware refinement is to let a
large `K*N` clear a slightly lower floor, not a higher one.

The advantage `Δ` itself also falls at small `M`, where the NPU underuses its array, so the
constant has a per-model component in principle. In practice that component is not
reachable through this knob. The next two sections say why, and what a candidate model
needs measured.

### The quantized floor's clamp

The knob cannot reach below 128, and that clamp sets the boundary. The function
`rocket_min_m_quant()` never returns less than `rocket_min_m()`, whose default is 128
[source-confirmed: `ggml-rocket.cpp`]. So `ROCKET_MIN_M_QUANT` values of 8, 32, 64, 96 and 128 are the same value, not
merely similar in effect. The admitted-GEMM count is identical at all of them, for every
model and clip length tested. It is 279 for SenseVoice and 24 or 217 for Parakeet, never
anything between [HW sweep]. A sweep that looks for the edge below 128 sweeps one point.

The operative gate at the boundary is therefore the F16 floor, not the quantized one, and a
single constant places every transition. The rule `M >= 128` predicts all three
transitions, on two models whose frame rates differ:

| family | rows | crosses 128 at | admitted below / above |
|---|---|---:|---|
| SenseVoice, all 279 GEMMs | `M = 16.6 s` | 7.8 s | 0 at 7 s, **279 at 8 s** |
| Parakeet, 193 of 217 | `M = 12.5 s` | 10.3 s | 24 at 10 s, **217 at 11 s** |
| Parakeet, the other 24 | `M = 2*12.5 s - 1` | 5.2 s | 0 at 5 s, 24 at 7 s |

The 8 s and 11 s cells test the constant: they ran after it was fitted on the other cells,
and both returned the predicted count. The 24-GEMM family is one GEMM per block, with
roughly twice the rows of the rest. That is the shape of a relative-position projection
over `2T-1` positions [hypothesis]. The row count is measured, and the attribution is not.

### The crossover

The crossover is a cliff, and there is no admitted-but-losing band. The table gives CPU
core-seconds freed, offload admitted against offload declined. Both arms run through the
same backend, so admission is the only variable. Each cell is the median of 3 warm reps,
with the first process and the first run of each discarded. The governor is pinned to
`performance`, and `scaling_cur_freq` reads back at 2352 MHz
[HW sweep, vendor RK1, A76-pinned, `aunt` conversational audio]:

| clip | 3 s | 5 s | 7 s | 10 s | 15 s | 20 s | 30 s | 45 s | 60 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SenseVoice-small 234M | 3.1% | -0.7% | 1.3% | **21.9%** | 31.0% | 34.4% | 42.5% | 45.0% | 43.1% |
| Parakeet-CTC 0.6B | 0.8% | 0.6% | 1.4% | 3.6% | **32.1%** | 40.0% | 43.4% | 46.0% | 46.8% |

The worst within-cell spread across the sweep is 11.9%, so every cell under about 12% is
unresolved. Read such a cell as "no measurable difference".

Wherever nothing is admitted, the two arms are indistinguishable. The first cell that
admits anything is already worth 22-32%. No length exists at which the offload is taken
and loses.

That settles where the floor belongs for this workload. The floor does not trade a small
win against a small loss at the margin. It is a switch, and the losing regime it would
exist to exclude is not reachable on either model. Any setting at or below 128 is the same
setting, and the relief saturates at 43-47%.

The wall does not move nearly as far. At 30 s it is 1.28x for SenseVoice and 1.33x for
Parakeet, against 42-43% of CPU freed. That is the distinction that §"Wall speedup and CPU
saving" draws.

Parakeet's two families can be priced apart, because a floor of 384 at 30 s admits the 24
and declines the 193. The 24 free 4.2% of the 43.4% that the full set frees. So 193 of 217
GEMMs carry roughly nine tenths of the benefit. The family that clears the floor earliest
is not where the value is.

## Short utterances below the F16 floor

A voice command is one to five seconds of audio. At 12.5 encoder frames a second, that is 16-75
rows. That is under the default `ROCKET_MIN_M` of 128, so nothing offloads and the NPU arm is the
CPU arm. With the floor lowered, every encoder GEMM goes to the NPU and re-packs its weight on
each call. The offload loses at no length measured. It ties the CPU at 1.3-1.8 s while
freeing a third of the CPU, and from 2 s up it is also faster.

Measured on an RK3588 (Turing RK1) on the mainline driver (`rocket` 1.3.0, kernel 7.2.8,
600 MHz), through whisper.cpp v1.9.4's `parakeet-cli`. The model is Parakeet TDT 0.6B v3 F16 on
4 threads pinned to the A76s, under the CPU governor `ondemand`. Each length is ten LibriSpeech
test-clean utterances. An utterance's cost is a 10-file process minus a 1-file process, over 9,
so no model load is in it. Three passes rotate the arm order, and the table is their medians,
every pass within 6% [HW sweep 2026-09-30]:

| audio | rows | CPU wall | NPU wall | wall | CPU core-s | NPU core-s | CPU freed |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.3-1.8 s | 16-23 | 516 ms | 520 ms | 1.00x | 2.01 | 1.32 | 34% |
| 2.1-2.5 s | 27-31 | 667 ms | 564 ms | 1.18x | 2.61 | 1.48 | 44% |
| 2.6-3.9 s | 33-49 | 938 ms | 673 ms | 1.41x | 3.71 | 1.73 | 54% |
| 4.4-5.9 s | 55-74 | 1449 ms | 838 ms | 1.73x | 5.73 | 2.21 | 61% |
| 7.1-8.9 s | 88-112 | 2278 ms | 1139 ms | 2.01x | 9.03 | 3.03 | 66% |
| 11.4-13.9 s | 143-174 | 3724 ms | 1659 ms | 2.25x | 14.77 | 4.51 | 69% |

The NPU arm runs `ROCKET_MIN_M=4`, and the row column is the audio at 12.5 frames a second. At
the default floor the NPU arm reads within 1% of the CPU through 5.9 s, because it admits no
GEMM there. It frees 4% at 7-9 s, where the 24 GEMMs of the `2T-1` family clear 128 rows, and
matches the forced arm from 11 s. Every arm's transcript is byte-identical to the CPU's over
three passes at every length. The word error against the references is 13.6, 1.9, 2.5, 0.7, 0.9
and 0.5% by length, and the shortest set's errors are proper names.

### The OpenMP wait policy

The CPU arm's user time is four threads times its wall, which reads like spin. It is work.
`OMP_WAIT_POLICY=PASSIVE` leaves the CPU arm's core-seconds within 1% at every length, and costs
it 2-5% of wall [HW sweep]. The same setting takes a further 7-17% off the NPU arm, which then
frees 45-72%. The NPU arm's threads do wait, on the NPU. So the relief is not the CPU arm's spin.

### A second process on the same cores

Freed core-seconds are worth something only to a process that can use them. A continuous stream
of 2.1-2.5 s utterances ran in one `parakeet-cli` process. Beside it ran `llama-bench` decode of
Llama-3.2-3B `Q4_K_M`, CPU only at 4 threads, both pinned to the same four A76s. The STT column
is the median utterance over the stream, against its uncontended cost above. The stream starts
4 s before the neighbor. Three passes, arms rotated [HW sweep 2026-09-30]:

| STT stream | LLM decode | against alone | STT per utterance | against uncontended |
|---|---:|---:|---:|---:|
| none | 7.07 t/s | 1.00x | n/a | n/a |
| CPU | 3.26 t/s | 0.46x | 1.19 s | 1.8x |
| NPU, floor 4 | 4.89 t/s | 0.69x | 2.39 s | 4.2x |
| NPU, floor 4, passive | 5.62 t/s | 0.79x | 1.98 s | 3.7x |

The offload gives the other process 1.5-1.7x the throughput, and the offloaded stream pays for
it in its own latency. Under contention the CPU stream slows 1.8x and the offloaded one
3.7-4.2x. So the offloaded utterance becomes the slower of the two. Each of an utterance's 265
offloaded GEMMs packs its weight and submits from a host thread. Each of those steps waits for a
core while the NPU idles [hypothesis].

An assistant that protects its listener deprioritizes the rest. The second run put the LLM at
`nice -n 19`, over two passes. The neighbor ran `-n 16 -r 2` beside an 800-utterance stream,
which outlived it in every arm [HW sweep 2026-09-30]:

| STT stream | LLM decode | against alone | STT per utterance | against uncontended |
|---|---:|---:|---:|---:|
| none | 7.23 t/s | 1.00x | n/a | n/a |
| CPU | 0.90 t/s | 0.12x | 0.78 s | 1.17x |
| NPU, floor 4 | 3.37 t/s | 0.47x | 1.54 s | 2.7x |
| NPU, floor 4, passive | 4.80 t/s | 0.66x | 0.89 s | 1.66x |

Priority protects the CPU stream and starves the LLM to an eighth, but it does not protect the
offloaded stream at the default wait policy. Under `OMP_WAIT_POLICY=PASSIVE`, the offloaded
stream costs the LLM a third of its rate, and the CPU stream costs it seven eighths. The price
is a 14% slower utterance, 0.89 s against 0.78. The likely cause is that the STT's own OpenMP
threads spin while its NPU workers wait for a core [hypothesis]. The passive policy puts them
to sleep. Both forced-arm windows are short, 8-10 utterances each, so their STT column is noisier.

### Where the floor comes from

The 128 default sits above an F16 crossover measured on LLM prefill. In that measurement,
Llama-3.2-3B read 0.35x at 16 rows and reached parity near 64. The current build does not
reproduce that table. The same model, NPU at floor 4 against the CPU, `llama-bench` at 4
threads, two passes within 2% [HW sweep 2026-09-30]:

| prompt tokens | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|
| NPU / CPU | 0.70x | 1.25x | 2.34x | 3.58x |

So on the current build the 3B crosses near 24 rows and Parakeet below 16. The 0.8B and 8B
rows of that prefill table are not re-measured on the current build, and a smaller model
crosses later.

### The per-call weight pack

Below 256 rows no weight goes resident. Of the 265 GEMMs an utterance runs, 120 also have no
stable weight name. So at these lengths every weight re-packs on every call. `ROCKET_MM_PROFILE`
puts `packB` at 0.59-0.60 thread-seconds an utterance at 1.3-2.5 s, against the NPU arm's
1.32-1.48 core-seconds [HW sweep]. A profile bucket is a thread interval. So that is an upper
bound on what a small-row resident route could remove, not its size.

### An int8 CPU runtime

The F16 CPU arm is not the fastest CPU path for this model. The runtime sherpa-onnx 1.13.8
runs Parakeet TDT 0.6B v3 as an int8 ONNX export on ONNX Runtime. The comparison is one
persistent process on the same four A76 threads and the same utterances. It ran three
passes, with the first dropped [HW sweep 2026-09-30]:

| audio | sherpa-onnx int8, CPU | F16, CPU | F16, NPU floor 4, passive |
|---|---:|---:|---:|
| 1.3-1.8 s | 231 ms, 0.92 core-s | 516 ms, 2.01 | 489 ms, 1.09 |
| 2.1-2.5 s | 290 ms, 1.15 | 667 ms, 2.61 | 537 ms, 1.24 |
| 2.6-3.9 s | 400 ms, 1.61 | 938 ms, 3.71 | 640 ms, 1.46 |
| 4.4-5.9 s | 768 ms, 2.69 | 1449 ms, 5.73 | 811 ms, 1.93 |
| 11.4-13.9 s | 1261 ms, 5.01 | 3724 ms, 14.77 | 1681 ms, 4.20 |

On a voice command the int8 runtime is about twice as fast as the NPU arm. It uses 7-16% less CPU
below 2.5 s, and 10-39% more from 2.6 s up. Its per-utterance walls are bimodal under `ondemand`:
the same 4.4-5.9 s utterance read 0.53 s in one pass and 1.02 s in another. A pinned governor
is expected to hold its fast mode [expected]. Its word error on the 1.3-1.8 s set is 27.3%
against the F16 model's 13.6%, over 22 words, so that difference is six errors against three.

### Setting `ROCKET_MIN_M`

`ROCKET_MIN_M=16` admits the same 265 GEMMs as a floor of 4 at every length measured, down to
1.3 s [HW sweep]. A floor of 32 drops the 1.3-2.5 s utterances to 24. A floor of 16 still keeps
whisper.cpp's beam-5 decode, 5 rows a step, on the CPU. Set it in the STT process only, because
the 3B loses at 16 rows (0.70x in §"Where the floor comes from"). Audio below 1.3 s is
unmeasured.

## Thread count

The thread count moves the wall and not the CPU. The host work is the same work whatever
number of threads carries it. So the thread count trades realtime headroom against latency
and leaves the CPU saving alone. The table is `base.en`, ten 3 s utterances, NPU arm
[HW sweep]:

| threads | CPU core-seconds | realtime | CPU arm core-seconds |
|---|---:|---:|---:|
| 1 | 41.96 | 0.76x | 66.25 |
| 2 | 39.03 | 1.42x | 67.60 |
| 4 | 40.39 | 2.36x | 67.67 |

The core-seconds are flat within 7% on the NPU arm and within 2% on the CPU arm, while the
wall moves 3.1x. The CPU saving holds at 37-42% at every thread count. **The realtime factor
is not flat**: at one thread `base.en` falls below realtime. So a caller who trades threads
away to protect other work on the box keeps the CPU saving. That caller can lose the
ability to keep up.

## Numerical faithfulness

Numerical faithfulness depends on the clip length and on the audio. It is not a yes/no
property. The NPU encoder is fp16 where the CPU's is fp32, so it is close but not
bit-identical. Whether that changes the transcript depends on how well-conditioned the
audio is. Both arms are deterministic (every repetition within an arm is byte-identical),
so a difference between arms is the numerics, not a race.

Word-level divergence between the arms, whisper.cpp [HW sweep]:

| audio | tiny.en | base.en | small.en |
|---|---|---|---|
| clean read speech, <= 60 s | identical | identical | identical |
| clean read speech, 120 s | diverges in a repetition loop | identical | identical |
| hard conversational, <= 10 s | identical | identical | identical |
| hard conversational, 30-60 s | identical | 4-6 word edits | identical |
| hard conversational, 120 s | identical | 66 edits of 169 words | 40 edits of 100 |

Neither arm is ground truth. On the hard clip both produce plausible readings of genuinely
ambiguous overlapping speech, and they differ in both directions. A perturbation early in
an autoregressive decode changes the prompt context for every window after it. So the
divergence compounds with length rather than staying local. **A transcript gate on clean
audio does not certify degraded audio**, and degraded audio is the case of a radio or
telephony caller.

On a CTC encoder the divergence does not compound with length, because there is no context
to carry it. The same offloaded-against-declined comparison ran on the two Q8_0 CTC models,
on hard conversational audio of 3-60 s [HW sweep]. Every cell that offloads nothing is
byte-identical. Every cell that offloads differs by one or two words in roughly
twenty-five. SenseVoice at 10 s reads `regaul` against `reaginaul`, and at 30 s reads `too`
against `two`. Parakeet at 30 s inserts a spurious `k`.

Neither arm is ground truth, and no reference transcript was scored. So the comparison
bounds the size of the disagreement and says nothing about which arm is better. It does
show a different shape from whisper's. Parakeet is byte-identical at 60 s while differing
at 30 s and 45 s, so the edits stay local and their count does not grow with the clip.
Determinism holds here too: across the whole sweep, no cell produced more than one distinct
transcript over its repetitions.

## Per-utterance cost with the model already loaded

A streaming caller loads the model once. The single-shot numbers in the earlier sections
carry a model load, which a server does not pay per utterance. The model load is 129 ms
for `tiny.en`, 173 ms for `base.en` and 333 ms for `small.en`. The table gives steady-state
CPU core-seconds per utterance on conversational audio. The harnesses are `whisper-cli` with
the clip repeated across `-f` arguments, and `transcribe-cli --batch`, which reuses one
context [HW sweep]:

| utterance | whisper tiny.en | whisper base.en | SenseVoice-small |
|---|---|---|---|
| 3 s | 3.03 -> 1.93 (36%) | 6.75 -> 3.82 (43%) | 1.34 -> 1.34 (0%) |
| 10 s | 3.39 -> 2.45 (28%) | 8.37 -> 4.89 (42%) | 4.67 -> 3.27 (30%) |
| 30 s | 4.38 -> 3.11 (29%) | 9.44 -> 6.17 (35%) | 13.37 -> 7.42 (44%) |

The two families cross. A CTC model pays what the audio costs, so it is much the cheapest
on a short utterance. At the default floor, that is exactly where the NPU does not reach
it.

Whisper pays a fixed 30 s encode whatever the utterance. That is wasteful in absolute CPU
on a 3 s clip, but it hands the NPU a job big enough to win every time. By 30 s, whisper
`base.en` on the NPU costs less absolute CPU than SenseVoice does. Whisper's cost is flat
in utterance length, and the CTC model's is linear.

## The weight cache on an F16 host

The weight cache that a streaming caller reaches for is not the cost on an F16 host. The
function `rocket_weight_key()` rejects ggml's positional autonames (`leaf_%d`). Those names
come from whisper.cpp, which never names its weight tensors, so every weight is re-packed
on every call rather than cached. For a streaming host that re-encodes against the same
weights every few seconds, a cache looks like the obvious lever. It is not, because the
per-call weight cost is small. The function `mm_pack_weights()` covers the prep, the
scatter and the fini, so `packB` is the whole per-call weight cost [HW sweep]:

| model | clip | packB | whole-process core-s | packB share |
|---|---|---:|---:|---:|
| tiny.en | 120 s | 43 ms | 19.8 | 0.22% |
| base.en | 120 s | 118 ms | 42.7 | 0.28% |
| small.en | 120 s | 551 ms | 97.0 | 0.57% |

The pack is at most 0.6% of the process CPU and at most ~4.5% of the encode wall. A cache
keyed on the weights buffer was built and measured on the mainline board. The keying works,
transcripts stay md5-identical, and packB goes to zero. The total gets 1.3% worse, because
the resident path's extra BO sync costs more than the pack it removes.

The F16 weight-residency result on LLM prefill (+18.5% at pp512, +13.6% at pp2048) does not
port to this workload. An LLM's weights are enormous against one micro-batch of
activations, while a whisper encoder's weights are small against 1500 rows of activations.

The host cost that does dominate is the per-worker input load. For `base.en` at 120 s, the
pack is 1244 ms, of which 103 ms is the shared A scatter and 118 ms is the weight scatter.
The rest is the per-worker copy of the input cube.

This cap is F16-only. With F16 weights, `weight_dequant` is 0 by construction. A Q8_0 host
pays it on every call, and a resident cache would remove the dequant and the scatter
together. That is a different term with a different share, and the next section measures
it.

## The weight cache on a Q8_0 host

On a Q8_0 host the cache's term is real, it is fixed per forward pass, and it is reachable
only across utterances. The removable term is `weight_dequant` + `packB`. On a quantized
host, at the short end, its share is an order of magnitude larger than the F16 cap.

The table covers two Q8_0 CTC encoders through `transcribe.cpp`, with the NPU arm at
`ROCKET_MIN_M_QUANT=128` (the floor that offloads for this workload). Each cell is the
median of two interleaved reps with a warm-up discarded, A76-pinned, under the CPU governor
`performance` [HW sweep 2026-08-26, vendor RK1]:

| model | clip | CPU arm core-s | NPU arm core-s | relief | dequant | packB | GEMM calls | cache cap (of NPU arm) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SenseVoiceSmall-Q8_0 | 10 s | 4.57 | 3.54 | 22.6% | 293 ms | 138 ms | 279 | **12.2%** |
| | 30 s | 13.59 | 8.01 | 41.1% | 264 ms | 152 ms | 279 | 5.2% |
| | 60 s | 32.12 | 18.41 | 42.7% | 246 ms | 173 ms | 279 | 2.3% |
| | 120 s | 86.74 | 54.28 | 37.4% | 249 ms | 160 ms | 279 | 0.8% |
| parakeet-ctc-0.6b-Q8_0 | 10 s | 8.97 | 8.41 | 6.2% | 29 ms | 27 ms | 24 | 0.7% |
| | 30 s | 25.62 | 14.36 | 43.9% | 584 ms | 538 ms | 217 | **7.8%** |
| | 60 s | 56.09 | 30.11 | 46.3% | 580 ms | 527 ms | 217 | 3.7% |
| | 120 s | 136.07 | 72.11 | 47.0% | 578 ms | 582 ms | 217 | 1.6% |

The term is fixed per forward pass, which is what the call column says. SenseVoice offloads
279 GEMMs whether the clip is 10 s or 120 s, and its dequant stays within 250-293 ms across
a 12x range of audio. Parakeet is the same from 30 s up, at 217. So the share falls with
clip length, from 12.2% down to 0.8%, and the lever is a short-utterance one.

Within one utterance there is nothing to cache. Those 279 calls are 279 distinct GEMMs, and
each dequantizes its own weight once. The count does not grow with audio length, so a
single-clip run has no reuse to exploit. The saving exists only across utterances, in a
process that keeps the model loaded. That is the steady-state regime that
§"Per-utterance cost with the model already loaded" measures, and the regime a streaming
service is in. A one-shot CLI invocation would gain nothing.

The cap is per model, not per family. Parakeet at 10 s offloads 24 GEMMs rather than 217
and buys 6.2% relief, because most of its layers sit under the floor at that length. Its
cache cap there is 0.7%. Measure the candidate model at the candidate utterance length.

## NPU clock parking on the vendor driver

The NPU clock does not park mid-encode on the vendor driver. Mainline `rocket` ships a
50 ms `autosuspend_delay_ms`, and host packing leaves gaps longer than that. So all three
cores idle inside a single encode, and the next submit runs at a third of the clock. The
patch `patches/rocket/081` raises the delay to 1000 ms.

The vendor `rknpu` driver sets `power_put_delay = 3000`, and its `rknpu_ondemand` governor
never measures load at all. So neither effect exists on the vendor driver. The clock was
sampled every 50 ms through one `small.en` encode: 231 samples, one distinct frequency,
zero transitions [HW sweep].
