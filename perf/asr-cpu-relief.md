# ASR on the NPU: what it frees, and the two floors that decide whether it frees anything

Speech-to-text is the first workload measured here in **CPU core-seconds** rather than in
throughput, because the question a streaming-ASR caller asks is not "how fast" but "how much
of the CPU do I get back". The two are different numbers and the difference is not small.

Measured on an RK3588 (Turing RK1) at 600 MHz on the **vendor `rknpu` 0.9.8** driver through
`rknpu-submit`, A76-pinned (`taskset -c 4-7`), 4 threads, `ROCKET_KACC=1`, CPU governor
`performance`. Both arms of every A/B come from one binary: for whisper.cpp the CPU arm is
`ROCKET_MIN_M=1000000 ROCKET_FLASH_ATTN=0` (every op declines at `supports_op`), for
transcribe.cpp it is `--backend cpu` against `--backend cpu_accel`. Warm-up run discarded,
arms interleaved with the order alternating. [HW sweep, 2026-08-24]

The raw sweeps, the harnesses and the clip generator are in [data/asr/](data/asr/).

## The wall speedup is not the CPU saving, and here it understates it

This backend moves the MACs to the NPU and leaves packA, packB and the output de-tile on the
host, and there is no on-chip layout conversion, so that host work is irreducible. The wall
therefore contains time the process spends *blocked* on the NPU with its threads idle, and
the CPU it frees is larger than the wall it saves.

whisper.cpp, `base.en`, clean speech: [HW sweep]

| clip | wall CPU | wall NPU | wall x | core-s CPU | core-s NPU | CPU freed |
|---|---:|---:|---:|---:|---:|---:|
| 3 s | 2.13 | 1.68 | 1.27x | 7.56 | 4.40 | **41.8%** |
| 30 s | 4.71 | 4.29 | 1.10x | 17.16 | 14.17 | **17.4%** |
| 120 s | 14.78 | 12.54 | 1.18x | 55.78 | 42.51 | **23.8%** |

A 1.10x wall at 30 s is a 17.4% CPU saving. Quoting the wall ratio as if it were the CPU
saving is wrong in both directions depending on the shape, so measure the one that is being
asked about.

## Whisper: the relief fraction rises with model size and falls with speech density

Every window is padded to 30 s whatever the audio, so the encoder's cost is fixed per window
and the decoder's is not. The encoder is the only part that offloads, so **the encoder's share
of the CPU core-seconds is the ceiling on the relief**, and that share is largest when the
model is big and the audio is sparse.

CPU core-seconds freed, clean speech / hard conversational speech: [HW sweep]

| model | 3 s | 10 s | 30 s | 60 s | 120 s |
|---|---:|---:|---:|---:|---:|
| tiny.en | 30.2 / 33.4% | 37.0 / 31.6% | 18.5 / 29.7% | 19.1 / 37.1% | 16.3 / 25.8% |
| base.en | 41.8 / 39.2% | 29.9 / 30.8% | 17.4 / 27.6% | 20.2 / 26.4% | 23.8 / 28.0% |
| small.en | 50.1 / 50.2% | 42.3 / 26.5% | 34.7 / 42.7% | 35.2 / 43.2% | 33.4 / 36.5% |

The NPU removes roughly **half** of the encoder's CPU cost, not all of it, because the packing
and the de-tile stay on the host. It never removes more than the encoder's share.

The **3 s column is the largest relief for every model**, and that is the padding working in
the NPU's favour: a 3 s clip pays a full 30 s encode and decodes almost nothing, so the
offloadable part is nearly the whole cost. A short-utterance workload sits at the favourable
end of this axis, not the unfavourable one.

Realtime factor, NPU arm, seconds of audio per second of wall: `tiny.en` 3-33x, `base.en`
1.8-12.9x, `small.en` 0.7-5.8x. The sub-realtime cells are the short clips, where the fixed
30 s encode and the model load dominate: 129 ms of model load for `tiny.en`, 173 ms for
`base.en` and 333 ms for `small.en`, paid per process.

## Two offload floors, and one of them is set for a different workload

An op reaches the NPU only above a row floor, and there are two:

- **`ROCKET_MIN_M` (default 128)** for F16/F32 weights. whisper's models are F16 and its
  encoder runs 1500 rows, so this never binds there. It binds on whisper's *decoder*, which
  is what it is for: `whisper-cli` beam search batches 5 decoders into one call, so every
  decode step is M=5, and a floor of 4 sends the whole decoder to the NPU at 2.3x slower than
  the CPU.
- **`ROCKET_MIN_M_QUANT` (default 512)** for quantized weights, which take a
  dequantize-to-fp16 path that needs more rows to amortize. **This floor is set from LLM
  prefill and it is the wrong floor for a CTC ASR encoder**, whose sequence length is the
  audio length in frames, a few hundred for a clip of a few tens of seconds.

The consequence is a step, not a ramp. CPU core-seconds freed against the CPU arm, Q8_0 CTC
models on conversational audio, three arms from one binary: [HW sweep]

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

At 30 s `ROCKET_MM_PROFILE` prints nothing at all for SenseVoice under the shipped floor: not
a small win, **zero offloaded ops**. At 60 s it prints 1395 job-batches. Lowering the floor
costs nothing anywhere it does not help (worst cell -0.6%) and above 60 s the two arms are
equal, because the sequence is over 512 either way.

So the "NPU buys nothing on short audio for these models" reading is two effects, not one, and
only the smaller one is a dispatch-cost property. **The knob accounts for everything down to
about 10 s; below that the win is genuinely gone.** At 3 s neither floor offloads usefully and
both arms are within 3%.

### A row floor is the right shape for it, and the weight size cancels

The temptation is to express the floor against the weight it is guarding: a quantized weight
costs more to decode, so a bigger weight should need more rows. It should not, and the profile
says why: the dequant a pass pays is **fixed in the row count**. SenseVoice offloads 279 GEMMs
and spends 250-293 ms decoding their weights whether the clip is 10 s or 120 s [HW sweep].

Write the criterion out. Offloading one GEMM wins when `M*K*N*Δ > K*N*d + F`, where `d` is the
per-element decode cost, `Δ` the per-MAC advantage over the CPU and `F` the fixed per-call
dispatch. Divide by `K*N`: **`M > d/Δ + F/(Δ*K*N)`**. The weight size divides out of the term
the floor exists to amortize, leaving a constant in `M` plus a correction that *shrinks* as the
weight grows. So a floor of the form `K*N*sizeof(weight)/M` would scale the threshold the wrong
way; a row floor is the right primary shape, and the shape-aware refinement available is to let
a large `K*N` clear a slightly lower floor, not a higher one.

`Δ` itself also falls at small `M`, where the NPU underuses its array, so the constant has a
per-model component in principle. In practice it is not reachable through this knob, and the
two sections below say why and what a candidate model actually needs measured.

### The knob cannot reach below 128, and that clamp is what sets the boundary

`rocket_min_m_quant()` is clamped to never return less than `rocket_min_m()`, whose default is
128 [source]. So `ROCKET_MIN_M_QUANT` values of 8, 32, 64, 96 and 128 are not merely similar in
effect, they are **the same value**, and the admitted-GEMM count is identical at all of them for
every model and clip length tested: 279 for SenseVoice, 24 or 217 for Parakeet, never anything
between [HW sweep]. A sweep looking for the edge below 128 is sweeping one point.

The operative gate at the boundary is therefore the F16 floor, not the quantized one, and a
single constant places every transition. `M >= 128` predicts all three, on two models whose
frame rates differ:

| family | rows | crosses 128 at | admitted below / above |
|---|---|---:|---|
| SenseVoice, all 279 GEMMs | `M = 16.6 s` | 7.8 s | 0 at 7 s, **279 at 8 s** |
| Parakeet, 193 of 217 | `M = 12.5 s` | 10.3 s | 24 at 10 s, **217 at 11 s** |
| Parakeet, the other 24 | `M = 2*12.5 s - 1` | 5.2 s | 0 at 5 s, 24 at 7 s |

The 8 s and 11 s cells were run to test the constant after it was fitted on the others, and both
returned the predicted count. The 24-GEMM family is one per block with roughly twice the rows of
the rest, which is the shape of a relative-position projection over `2T-1` positions
**[hypothesis]**; the row count is measured, the attribution is not.

### The crossover is a cliff, and there is no admitted-but-losing band

CPU core-seconds freed, offload admitted against offload declined through the same backend so
that admission is the only variable, medians of 3 warm reps with the first process and the first
run of each discarded, governor pinned to `performance` and `scaling_cur_freq` read back at
2352 MHz [HW sweep, vendor RK1, A76-pinned, `aunt` conversational audio]:

| clip | 3 s | 5 s | 7 s | 10 s | 15 s | 20 s | 30 s | 45 s | 60 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SenseVoice-small 234M | 3.1% | -0.7% | 1.3% | **21.9%** | 31.0% | 34.4% | 42.5% | 45.0% | 43.1% |
| Parakeet-CTC 0.6B | 0.8% | 0.6% | 1.4% | 3.6% | **32.1%** | 40.0% | 43.4% | 46.0% | 46.8% |

Worst within-cell spread across the sweep is 11.9%, so every cell under about 12% is
unresolved and should be read as "no measurable difference".

**Wherever nothing is admitted the two arms are indistinguishable, and the first cell that
admits anything is already worth 22-32%.** There is no length at which the offload is taken and
loses. That is what closes the question of where the floor belongs for this workload: the floor
is not trading a small win against a small loss at the margin, it is a switch, and the losing
regime it would exist to exclude is not reachable on either model. Anything at or below 128 is
the same setting, and the relief saturates at 43-47%.

The wall does not move nearly as far, 1.28x for SenseVoice and 1.33x for Parakeet at 30 s
against 42-43% of CPU freed, which is the same distinction the top of this page draws.

Parakeet's two families can be priced apart, because a floor of 384 at 30 s admits the 24 and
declines the 193. The 24 free **4.2%** of the 43.4% the full set frees, so 193 of 217 GEMMs
carry roughly nine tenths of the benefit and the family that clears the floor earliest is not
where the value is.

## Thread count moves the wall and not the CPU

The host work is the same work whatever it is spread across, so the thread count trades realtime
headroom against latency and leaves the deliverable alone. `base.en`, ten 3 s utterances, NPU
arm: [HW sweep]

| threads | CPU core-seconds | realtime | CPU arm core-seconds |
|---|---:|---:|---:|
| 1 | 41.96 | 0.76x | 66.25 |
| 2 | 39.03 | 1.42x | 67.60 |
| 4 | 40.39 | 2.36x | 67.67 |

Flat within 7% on the NPU arm and within 2% on the CPU arm, while the wall moves 3.1x. The CPU
saving holds at 37-42% at every thread count. **The realtime factor is not flat**, and at one
thread `base.en` falls below realtime, so a caller trading threads away to protect other work on
the box keeps the CPU saving and can lose the ability to keep up.

## Numerical faithfulness is length- and audio-dependent, not a yes/no

The NPU encoder is fp16 where the CPU's is fp32, so it is close but not bit-identical, and
whether that changes the transcript depends on how well-conditioned the audio is. Both arms
are **deterministic** (every repetition within an arm is byte-identical), so a difference
between arms is the numerics, not a race.

Word-level divergence between the arms, whisper.cpp: [HW sweep]

| audio | tiny.en | base.en | small.en |
|---|---|---|---|
| clean read speech, <= 60 s | identical | identical | identical |
| clean read speech, 120 s | diverges in a repetition loop | identical | identical |
| hard conversational, <= 10 s | identical | identical | identical |
| hard conversational, 30-60 s | identical | 4-6 word edits | identical |
| hard conversational, 120 s | identical | 66 edits of 169 words | 40 edits of 100 |

Neither arm is ground truth: on the hard clip both produce plausible readings of genuinely
ambiguous overlapping speech, and they differ in both directions. The pattern is that a
perturbation early in an autoregressive decode changes the prompt context for every window
after it, so the divergence compounds with length rather than staying local. **A transcript
gate on clean audio does not certify degraded audio**, which is the case a radio or telephony
caller is actually in.

**On a CTC encoder the divergence does not compound with length, because there is no context to
carry it.** The same offloaded-vs-declined comparison on the two Q8_0 CTC models, hard
conversational audio, 3-60 s [HW sweep]: every cell that offloads nothing is byte-identical, and
every cell that offloads differs by one or two words in roughly twenty-five: SenseVoice at 10 s
reads `regaul` against `reaginaul`, at 30 s `too` against `two`; Parakeet at 30 s inserts a
spurious `k`. Neither arm is ground truth and no reference transcript was scored, so this bounds
the size of the disagreement and says nothing about which is better. What it does show is a
different shape from whisper's: Parakeet is byte-identical at 60 s while differing at 30 s and
45 s, so the edits stay local and their count does not grow with the clip. Determinism holds
here too: across the whole sweep no cell produced more than one distinct transcript over its
repetitions.

## Per-utterance cost with the model already loaded

A streaming caller loads the model once, so the single-shot numbers above carry a model load
(129 / 173 / 333 ms for `tiny.en` / `base.en` / `small.en`) that a server does not pay per
utterance. Steady state, conversational audio, CPU core-seconds **per utterance**
(`whisper-cli` with the clip repeated across `-f` arguments; `transcribe-cli --batch`, which
reuses one context): [HW sweep]

| utterance | whisper tiny.en | whisper base.en | SenseVoice-small |
|---|---|---|---|
| 3 s | 3.03 -> **1.93** (36%) | 6.75 -> **3.82** (43%) | 1.34 -> 1.34 (0%) |
| 10 s | 3.39 -> **2.45** (28%) | 8.37 -> **4.89** (42%) | 4.67 -> **3.27** (30%) |
| 30 s | 4.38 -> **3.11** (29%) | 9.44 -> **6.17** (35%) | 13.37 -> **7.42** (44%) |

The two families cross. A CTC model pays what the audio costs, so it is much the cheapest on a
short utterance, and that is exactly where the NPU cannot help it. Whisper pays a fixed 30 s
encode whatever the utterance, which is wasteful in absolute CPU on a 3 s clip but hands the
NPU a job big enough to win every time. By 30 s whisper `base.en` on the NPU costs **less**
absolute CPU than SenseVoice does, because whisper's cost is flat in utterance length and the
CTC model's is linear.

## The weight cache a streaming caller reaches for is not the cost, and it was built once

`rocket_weight_key()` rejects ggml's positional autonames (`leaf_%d`), which whisper.cpp
produces because it never names its weight tensors, so every weight is re-packed on every
call rather than cached. For a streaming host that re-encodes against the same weights every
few seconds this looks like the obvious lever. It is not: `mm_pack_weights()` covers the
prep, the scatter and the fini, so `packB` **is** the whole per-call weight cost, and it is
[HW sweep]

| model | clip | packB | whole-process core-s | packB share |
|---|---|---:|---:|---:|
| tiny.en | 120 s | 43 ms | 19.8 | 0.22% |
| base.en | 120 s | 118 ms | 42.7 | 0.28% |
| small.en | 120 s | 551 ms | 97.0 | 0.57% |

At most **0.6% of the process CPU** and at most ~4.5% of the encode wall. The lever was built
and run once already, on the mainline board: keying on the weights buffer works, transcripts stay
md5-identical, packB goes to zero, and the total gets **1.3% worse**, because the resident path's
extra BO sync costs more than the pack it removes. The prior that made
this look worth building is the F16 weight-residency result on LLM prefill (+18.5% / +13.6%
at pp512 / pp2048), and it does not port: an LLM's weights are enormous against one
micro-batch of activations, while a whisper encoder's are small against 1500 rows of them.
The host cost that *does* dominate is the per-worker input load: for `base.en` at 120 s,
1244 ms of pack of which 103 ms is the shared A scatter and 118 ms is the weight scatter, the
rest being the per-worker copy of the input cube.

**This cap is F16-only, and that is not incidental**: with F16 weights `weight_dequant` is 0 by
construction. A Q8_0 host pays it on every call, and a resident cache would remove the dequant
and the scatter together. That is a different term with a different share, and it is measured
below.

## On a Q8_0 host the cache's term is real, fixed per pass, and only reachable across utterances

The removable term is `weight_dequant` + `packB`, and on a quantized host it is an order of
magnitude larger a share than the F16 cap above, at the short end. Two Q8_0 CTC encoders
through `transcribe.cpp`, NPU arm at `ROCKET_MIN_M_QUANT=128` (the floor that actually offloads
for this workload), medians of two interleaved reps with a warm-up discarded, A76-pinned, CPU
governor `performance` [HW sweep 2026-08-26, vendor RK1]:

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

**The term is fixed per forward pass**, which is what the call column says: SenseVoice offloads
279 GEMMs whether the clip is 10 s or 120 s, and its dequant stays within 250-293 ms across a
12x range of audio. Parakeet is the same from 30 s up at 217. So the share falls with clip
length, 12.2% down to 0.8%, and the lever is a **short-utterance** one.

**Within one utterance there is nothing to cache.** Those 279 calls are 279 distinct GEMMs, each
dequantizing its own weight once; the count does not grow with audio length, so a single-clip run
has no reuse to exploit. The saving exists only across utterances in a process that keeps the
model loaded, which is the steady-state regime the per-utterance table above measures, and the
regime a streaming service is in. A one-shot CLI invocation would gain nothing.

**The cap is per model, not per family.** Parakeet at 10 s offloads 24 GEMMs rather than 217
and buys 6.2% relief, because most of its layers sit under the floor at that length; its cache
cap there is 0.7%. Measure the candidate model at the candidate utterance length.

## The clock does not park mid-encode on the vendor driver

Mainline `rocket` ships a 50 ms `autosuspend_delay_ms`, and host packing leaves gaps longer
than that, so all three cores idle inside a single encode and the next submit runs at a third
of the clock; `patches/rocket/081` raises it to 1000 ms. The vendor `rknpu` driver sets
`power_put_delay = 3000` and its `rknpu_ondemand` governor never measures load at all, so
neither effect exists there. Sampled every 50 ms through one `small.en` encode: 231 samples,
one distinct frequency, zero transitions. [HW sweep]
