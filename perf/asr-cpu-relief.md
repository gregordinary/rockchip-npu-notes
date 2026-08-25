# ASR on the NPU: what it frees, and the two floors that decide whether it frees anything

Speech-to-text is the first workload measured here in **CPU core-seconds** rather than in
throughput, because the question a streaming-ASR caller asks is not "how fast" but "how much
of the CPU do I get back". The two are different numbers and the difference is not small.

Measured on an RK3588 (Turing RK1) at 600 MHz on the **vendor `rknpu` 0.9.8** driver through
`rknpu-submit`, A76-pinned (`taskset -c 4-7`), 4 threads, `ROCKET_KACC=1`, CPU governor
`performance`. Both arms of every A/B come from one binary — for whisper.cpp the CPU arm is
`ROCKET_MIN_M=1000000 ROCKET_FLASH_ATTN=0` (every op declines at `supports_op`), for
transcribe.cpp it is `--backend cpu` against `--backend cpu_accel`. Warm-up run discarded,
arms interleaved with the order alternating. [HW sweep, 2026-08-24]

The raw sweeps, the harnesses and the clip generator are in [data/asr/](data/asr/).

## The wall speedup is not the CPU saving, and here it understates it

This backend moves the MACs to the NPU and leaves packA, packB and the output de-tile on the
host — there is no on-chip layout conversion, so that host work is irreducible. The wall
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
of the CPU core-seconds is the ceiling on the relief** — and that share is largest when the
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
30 s encode and the model load dominate — 129 / 173 / 333 ms of model load for
tiny / base / small, paid per process.

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
  audio length in frames — a few hundred for a clip of a few tens of seconds.

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

At 30 s `ROCKET_MM_PROFILE` prints nothing at all for SenseVoice under the shipped floor — not
a small win, **zero offloaded ops**. At 60 s it prints 1395 job-batches. Lowering the floor
costs nothing anywhere it does not help (worst cell -0.6%) and above 60 s the two arms are
equal, because the sequence is over 512 either way.

So the "NPU buys nothing on short audio for these models" reading is two effects, not one, and
only the smaller one is a dispatch-cost property. **The knob accounts for everything down to
about 10 s; below that the win is genuinely gone** — at 3 s neither floor offloads usefully and
both arms are within 3%.

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
thread `base.en` falls below realtime — so a caller trading threads away to protect other work on
the box keeps the CPU saving and can lose the ability to keep up.

## Numerical faithfulness is length- and audio-dependent, not a yes/no

The NPU encoder is fp16 where the CPU's is fp32, so it is close but not bit-identical, and
whether that changes the transcript depends on how well-conditioned the audio is. Both arms
are **deterministic** — every repetition within an arm is byte-identical — so a difference
between arms is the numerics, not a race.

Word-level divergence between the arms, whisper.cpp: [HW sweep]

| audio | tiny.en | base.en | small.en |
|---|---|---|---|
| clean read speech, ≤ 60 s | identical | identical | identical |
| clean read speech, 120 s | diverges in a repetition loop | identical | identical |
| hard conversational, ≤ 10 s | identical | identical | identical |
| hard conversational, 30-60 s | identical | 4-6 word edits | identical |
| hard conversational, 120 s | identical | 66 edits of 169 words | 40 edits of 100 |

Neither arm is ground truth: on the hard clip both produce plausible readings of genuinely
ambiguous overlapping speech, and they differ in both directions. The pattern is that a
perturbation early in an autoregressive decode changes the prompt context for every window
after it, so the divergence compounds with length rather than staying local. **A transcript
gate on clean audio does not certify degraded audio**, which is the case a radio or telephony
caller is actually in.

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
short utterance — and that is exactly where the NPU cannot help it. Whisper pays a fixed 30 s
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
md5-identical, packB goes to zero — and the total gets **1.3% worse**, because the resident path's
extra BO sync costs more than the pack it removes. The prior that made
this look worth building is the F16 weight-residency result on LLM prefill (+18.5% / +13.6%
at pp512 / pp2048), and it does not port: an LLM's weights are enormous against one
micro-batch of activations, while a whisper encoder's are small against 1500 rows of them.
The host cost that *does* dominate is the per-worker input load — for `base.en` at 120 s,
1244 ms of pack of which 103 ms is the shared A scatter and 118 ms is the weight scatter, the
rest being the per-worker copy of the input cube.

**This cap is F16-only, and that is not incidental**: with F16 weights `weight_dequant` is 0 by
construction. A Q8_0 host pays it on every call — 254 ms in one SenseVoice 60 s profile against a
1756 ms pack — and a resident cache would remove the dequant and the scatter together. That is a
different term with a different share and nothing here measures it.

## The clock does not park mid-encode on the vendor driver

Mainline `rocket` ships a 50 ms `autosuspend_delay_ms`, and host packing leaves gaps longer
than that, so all three cores idle inside a single encode and the next submit runs at a third
of the clock — `patches/rocket/081` raises it to 1000 ms. The vendor `rknpu` driver sets
`power_put_delay = 3000` and its `rknpu_ondemand` governor never measures load at all, so
neither effect exists there. Sampled every 50 ms through one `small.en` encode: 231 samples,
one distinct frequency, zero transitions. [HW sweep]
