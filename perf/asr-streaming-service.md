# A whisper.cpp transcription service on the NPU

A streaming caller does not run `whisper-cli`. OpenWebRX+ (1.2.123) keeps `whisper-server`
loaded and posts it a chunk of squelch-gated audio every `chunkSeconds` of wall time. The
default is 20 s, as one 12 kHz WAV with no other request fields. So the server's command-line
defaults decide every decode parameter. Every request costs one full 30 s-padded encode however
long the chunk is. The number the caller pays is CPU core-seconds per second of audio while the
channel is busy.

This page measures that number and the levers that move it, on `ggml-small` (multilingual, F16),
the model that deployment runs. Three configurations come out of it, in the table at the end.
The cheapest keeps the client's 20 s latency and returns 30% of the CPU at better accuracy than
the defaults.

Measured on an RK3588 (Turing RK1) at 600 MHz, mainline `rocket` 1.3.0, A76-pinned
(`taskset -c 4-7`), 4 threads, CPU governor `performance`, `ROCKET_KACC=1`, whisper.cpp master
`eacbd82`. Both arms come from one binary. The CPU arm declines every op at `supports_op`
(`ROCKET_MIN_M=1000000 ROCKET_FLASH_ATTN=0`). The server process is billed by its
`utime + stime` sampled around each request, so the model load is outside every number. Arms
rotate across two passes with a memory reset before each arm, and ratios are paired within a
pass. [HW sweep 2026-09-03 and 2026-09-04]

The harness, the chunk sets and the raw sweeps are in [data/asr-service/](data/asr-service/).

## The request

A 20 s request on the shipped NPU arm is 5.3 s of wall and 17.3 CPU core-seconds. The CPU arm
is 7.8 s and 29.7. Per second of audio that is **1.06 against 1.74 core-seconds**, so the NPU
frees 39% of the CPU before any tuning. The table is where those core-seconds go, by `perf`
sample over the NPU arm's process, on a 20 s greedy window. [HW sweep]

| Term | Share of CPU samples | Where it runs |
|---|---:|---|
| Encoder self-attention (`flash_attn_ext_tiled`, `flash_attn_ext`, `vec_soft_max`) | 54% | CPU. The op carries no mask and the gate declined it |
| Decoder weight GEMV (`ggml_vec_dot_f16`), ~80 steps of M=1 per 20 s | 21% | CPU, by design |
| Other ggml ops (GELU, LayerNorm, adds, conv1) | ~6% | CPU |
| Backend and driver: input scatter, weight scatter, de-tile | 3.4% | CPU |
| Mel spectrogram (`fft`, `log_mel_spectrogram`) | 2.4% | CPU |
| libgomp barriers | 2.7% | CPU |

The encoder's twelve `MUL_MAT`s per layer and the cross-attention K/V projections are what
offload, 480 job-batches per request. The encoder is 3.0 s of the 5.3 s wall. The two largest
CPU terms are therefore attention the NPU does not compute and a decoder it cannot. Every
lever below is priced against that table.

## The encoder attention offload

whisper's encoder attention is unmasked, and the `FLASH_ATTN_EXT` handler takes a NULL mask as
unmasked. Admitting the op is a two-line gate change, `ROCKET_FLASH_ATTN_UNMASKED=1`. It
computes the same transcript (md5-identical) and buys nothing. [HW sweep, 20 s greedy
`whisper-cli`, two reps]

| | CPU core-seconds | Encode wall |
|---|---:|---:|
| Attention on the CPU (shipped) | 16.2 | 3.17 s |
| Attention on the NPU, host softmax | 16.1 | 4.31 s |
| Attention on the NPU, on-chip softmax (`ROCKET_ATTN_HOST_SOFTMAX=0`) | 20.3 | 6.20 s |

The 36.5% of samples the CPU kernel held moved, to the sample, into the offload's host half.
Scalar `expf` took 17.0%, `host_softmax_rows` 6.9%, `fa_mask_scores` 5.4%, the batch runner
4.7% and the score de-tile 2.1%.

At head size 64 the `[T, n_kv]` score matrix per head is 2.3M
elements. It is moved out to the host, exponentiated one at a time, and moved back. That costs
what the QK and AV MACs cost the CPU kernel. Score traffic per MAC is `1/head_dim`, so
the tax halves at an LLM's 128 [expected]. The knob ships off, and this is the whole reason.

## The decoder's re-runs

Three things in `whisper_full` multiply a request's decode cost. All three are decided by the
server's defaults, because the client sends no fields [source-confirmed, whisper.cpp `eacbd82`]:

- The temperature ladder. A window whose decode fails the entropy threshold (2.4) or the
  log-probability threshold (-1.0) is decoded again at temperature 0.2. The ladder continues
  to 1.0 in steps of 0.2, with two decoders from the second rung on. Each rung is a full
  decode of the window.
- The timestamp failures. With timestamps on, a decoder also fails when a segment's end
  timestamp moves backwards. It fails again when it produces no timestamped segment. Both
  failures take the same ladder.
- The re-window. When the last segment ends before the chunk's end, `seek` advances to that
  end. The remainder is then encoded and decoded as a new window, padded to 30 s again.

`-nt` (`no_timestamps`) removes the second and the third. On completion the window advances by
the full 30 s, whatever the decoder emitted. The ladder itself is controlled by
`temperature_inc`, and here `whisper-server` differs from `whisper-cli`. The server parses
`-nf` and never applies it, so a server started with `-nf` still climbs the ladder. The
per-request field `temperature_inc=0` is what turns it off today. The one-line server fix is in
[data/asr-service/whisper-server-no-fallback.patch](data/asr-service/whisper-server-no-fallback.patch).

What a rung costs: one clean 20 s chunk trips the ladder in every arm. It reads 16.6 s of wall
and 55.6 core-seconds on the NPU arm, against 5.3 s and 17.3 for a normal one, a factor of 3.2.
Over the 20 s sweep the shipped arm reports 7 failures in 36 requests. `-ac 1000` alone raises
that to 31, which is why its encode saving reaches the CPU bill only in part. [HW sweep]

The re-window is what turns a longer chunk into a loss. A 30 s chunk fills the window, so it is
cut mid-word at the window's edge. The decoder's last timestamp lands short of the edge, and
the remainder is encoded again. That is 48 encodes for 26 requests on the 30 s arm, against
39 for 38 on the 20 s arm.

The 30 s arm therefore costs 13% more per second of audio than the 20 s arm. A 20 s chunk's
own silence padding lets the decoder reach the end of the audio. A 30 s chunk needs `-nt` to be
one encode. [HW sweep]

Turning the ladder off (`temperature_inc=0`) on the shipped 20 s configuration reads 0.832x on
clean speech, 0.998x at 15 dB and 0.919x at 5 dB. The chunks that trip it are not the noisy
ones: the 15 dB set never tripped it, and the clean set's one hard chunk did. It costs 1.5 WER
points on clean speech and 6 at 5 dB, so the ladder was buying accuracy on the chunks it
re-decoded. [HW sweep]

## The audio context

`-ac N` runs the encoder over the first N of its 1500 positions, 50 per second, and the stream
example ships it as a speed knob. Set to the chunk's own length it halves the encode and breaks
the decoder in a specific way. Every chunk transcribes correctly up to the audio's end and then
continues into a loop, "unnotting, unnotting, unnotting", until the token limit. The model
never sees the silence that closes an utterance, because the context ends where the speech
does. The temperature ladder catches some of those loops as repetition failures and re-decodes
them. That is why `-ac 1000` alone reads 21.0% WER on clean speech and `-ac 1000` with the
ladder off reads 26.8%, against 8.9% for the full context. [HW sweep]

With timestamps off as well, nothing bounds the run-on. `-ac 1000 -nt` with the ladder off
decodes every chunk to the 448-token window limit. It reads 103% WER on clean speech and 110%
at 15 dB, more inserted words than the reference has. So the timestamp machinery was limiting
the damage, and `-nt` is safe only where the context is not the problem. [HW sweep]

The remedy is the same rule the chunk length obeys: leave room after the audio. `-ac 1200` on
a 20 s chunk is 4 s of context past the cut. It transcribes every chunk end cleanly, trips no
fallback, and encodes in 2.25 s against 3.07 s.

With `-nt` it reads 0.636 / 0.775 / 0.665 of
the shipped arm's CPU. Its WER is 9.3 / 15.2 / 64.6 against 8.9 / 16.7 / 68.7. The rule is
`audio_ctx = 50 x (chunk seconds + 4)`, capped at 1500. A 30 s chunk has no headroom to give,
which is why it takes the `-nt` route instead. [HW sweep]

## Levers, one at a time

Every arm is the shipped NPU arm plus one change. The unit is CPU core-seconds per second of
audio, paired against the shipped arm within a pass. The two passes agree to three decimals and
the same-configuration repeat reads 0.999, so a difference under 1% is nothing.

WER is the word
error rate of the concatenated transcript against the LibriSpeech reference. Its absolute level
carries the mid-word cuts every chunk boundary makes and the chapter's archaic names. Read it
across arms rather than as a claim about the model. [HW sweep, 2 passes]

| Arm | CPU, clean | CPU, 15 dB | CPU, 5 dB | Request wall | WER clean / 15 dB / 5 dB |
|---|---:|---:|---:|---:|---|
| CPU only | 1.645 | 1.730 | 1.813 | +42 to +54% | 8.9 / 16.8 / 66.6 |
| NPU, shipped (the reference) | 1.000 | 1.000 | 1.000 | 6.5 s / 5.3 s / 5.1 s | 8.9 / 16.7 / 68.7 |
| `-ac 1000` | 0.860 | 0.892 | 0.773 | -9 to -21% | 21.0 / 17.9 / 59.6 |
| `-nt` | 0.782 | 0.955 | 0.916 | -5 to -21% | 9.5 / 15.8 / 62.4 |
| `-t 2` | 0.874 | 0.877 | 0.883 | +51% | 8.9 / 16.7 / 68.7 |
| `ggml-small-q5_1` | 0.789 | 0.947 | 1.044 | -20 to +4% | 11.3 / 15.6 / 67.2 |
| `ggml-small-q8_0` | 1.056 | 1.095 | 1.137 | -1 to -7% | 9.9 / 17.3 / 70.7 |
| 30 s chunks | 0.992 | 1.205 | 1.183 | one request per 30 s | 5.8 / 14.0 / 67.8 |
| unpinned (no `taskset`) | 1.024 | 1.021 | 1.023 | +7% | 8.9 / 16.7 / 68.7 |
| encoder attention on the NPU | 1.003 | 0.985 | 1.161 | +20 to +38% | 8.9 / 16.8 / 66.3 |

What each row says:

- **The NPU is worth 39-45% of the CPU before any tuning.** The relief is largest on the
  noisiest audio, where the decoder emits least.
- **`-ac 1000` halves the encode** (3.07 to 1.65 s per request) and returns only 16% of the
  CPU. The shorter context trips the fallback ladder 31 times in 36 requests against 7, and
  each trip is a full re-decode. On clean speech it costs 12 WER points, on the 5 dB channel it
  gains 9. The loops in the section above are the reason.
- **`-nt` is the one free lever**: 12% of the CPU at equal or better WER. It removes the
  timestamp failure branches (entropy failures 3 to 0), the re-window (encodes 39 to 36) and
  17% of the decoded tokens.
- **`-t 2` trades latency for CPU** at a fixed rate: 12% fewer core-seconds for a request half
  again as long, and the transcript is bit-identical. The realtime factor falls from 3.1-3.9x
  to 2.1-2.6x, still above the channel.
- **A quantized model does not pay here.** Q8_0 costs 10% more CPU than F16. The decoder step
  is 20% faster, but the encoder's weights are dequantized to fp16 on every request, about 3
  core-seconds. Its unnamed tensors keep them off the resident route. Q5_1 reads
  half as many bytes in that dequant and lands at -7%, at +2.4 WER points on clean speech.
- **30 s chunks cost 13% more per second of audio**, the re-window above. They improve WER by
  three points on clean speech and at 15 dB, because there are fewer cuts.
- **Pinning to the A76 cluster is worth 2%** under the `performance` governor. The bias that
  pinning corrects elsewhere is the load-sampling governor's, and that is measured separately
  below.
- **The attention offload is parity at best**, and the reason is in its own section above.

## Levers stacked

The same unit, against the same reference. The ladder is off in every row (`temperature_inc=0`).
[HW sweep, 2 passes]

| Arm | CPU, clean | CPU, 15 dB | CPU, 5 dB | Encodes per request | WER clean / 15 dB / 5 dB |
|---|---:|---:|---:|---:|---|
| 20 s, ladder off | 0.832 | 0.998 | 0.919 | 1.00 | 10.4 / 16.7 / 74.8 |
| 20 s, `-nt`, ladder off | 0.782 | 0.955 | 0.838 | 1.00 | 9.5 / 15.8 / 68.9 |
| 20 s, `-ac 1200`, ladder off | 0.710 | 0.812 | 0.770 | 1.03 | 8.9 / 16.1 / 63.1 |
| 20 s, `-ac 1200 -nt`, ladder off | 0.636 | 0.775 | 0.665 | 1.00 | 9.3 / 15.2 / 64.6 |
| 20 s, `-ac 1200 -nt -sns`, ladder off | 0.638 | 0.774 | 0.689 | 1.00 | 7.7 / 13.8 / 63.6 |
| 25 s chunks, defaults | 0.756 | 0.981 | 1.146 | 1.10 | 6.8 / 21.4 / 52.2 |
| 25 s chunks, ladder off | 0.756 | 0.981 | 1.281 | 1.17 | 6.8 / 21.4 / 96.2 |
| 25 s chunks, `-nt`, ladder off | 0.710 | 0.852 | 0.825 | 1.00 | 8.3 / 15.0 / 56.4 |
| 25 s chunks, `-ac 1250`, ladder off | 0.828 | 1.034 | 0.970 | 1.57 | 10.7 / 17.8 / 77.2 |
| 30 s chunks, `-nt`, ladder off | 0.639 | 0.741 | 0.557 | 1.00 | 5.3 / 19.0 / 79.5 |
| 20 s, `-ac 1000`, ladder off | 0.623 | 0.817 | 0.651 | 1.08 | 26.8 / 20.3 / 61.6 |
| 20 s, `-ac 1000 -nt`, ladder off | 0.797 | 1.069 | 0.532 | 1.00 | 103 / 110 / 75 |
| the same, `-t 2` | 0.636 | 0.847 | 0.460 | 1.00 | 101 / 110 / 75 |
| the same, `ggml-small-q5_1` | 0.816 | 0.938 | 0.563 | 1.00 | 142 / 109 / 73 |

The three `-ac 1000` rows are the run-on, and their CPU columns are the cost of decoding loops
rather than of transcribing. The 25 s chunk with defaults removes most of the re-window, 1.10
encodes per request against 1.85 at 30 s, and lands near parity anyway. It costs the 15 dB set
five WER points, and one 5 dB chunk decoded a 151-token loop without a ladder to interrupt it.
The same chunk with `-nt` is the length done right. It encodes once per request, returns 20%
of the CPU, and beats the shipped arm's WER on every condition, at 25 s of latency.

`-sns` (suppress non-speech tokens) adds nothing to the CPU and takes a point or more off the
WER on every condition. On audio it cannot parse, whisper emits a run of bracket tokens, and
those are the tokens `-sns` suppresses.

## Word error rate on a radio-shaped channel

The absolute level of the WER column is set by the chunk cuts and by the chapter rather than by
the model. Every 20 s boundary falls mid-word. The reference is an archaic text full of names
(`Montfichet`, `Fitzooth`) the model spells its own way. Read the column across arms.

On the clean set every arm that keeps its transcript sits between 7.7 and 9.5 points. The 30 s
chunks reach 5.3 because there are fewer cuts. On the 15 dB set the range is 13.8 to 16.7, and
on the 5 dB set 56 to 69. The NPU arm and the CPU arm agree on clean speech and differ by two
points at 5 dB. That is the fp16 encoder on audio the decoder is already guessing at.

The 5 dB set is whisper's hallucination regime, and its CPU numbers say more about that than
about any lever. A chunk the model cannot parse comes back as a run of bracket tokens. One 25 s
chunk at 5 dB decoded a 151-token loop that cost 26 s of wall and 86 core-seconds, five times
a normal request.

The temperature ladder is the only thing in the default configuration that
interrupts such a loop. That is why turning it off on that set can cost rather than save.
With a 20 dB squelch in front of the transcriber the 15 dB set is the representative one. A
noise burst inside an open squelch is the case the 5 dB set stands for. [HW sweep]

The channel is a proxy. It is band-passed, noisy, offset and fading read speech with a known
transcript. What it cannot certify is real fading, real compression and real accents. The
WER column ranks configurations against each other rather than predicting a channel.

## The operating point

Three configurations survive with their transcript intact, and they differ in latency:

| | Server | Request field | CPU (clean / 15 dB / 5 dB) | WER clean / 15 dB / 5 dB |
|---|---|---|---|---|
| 20 s chunks | `-nt -ac 1200 -sns` | `temperature_inc=0` | 0.638 / 0.774 / 0.689 | 7.7 / 13.8 / 63.6 |
| 25 s chunks | `-nt` | `temperature_inc=0` | 0.710 / 0.852 / 0.825 | 8.3 / 15.0 / 56.4 |
| 30 s chunks | `-nt` | `temperature_inc=0` | 0.639 / 0.741 / 0.557 | 5.3 / 19.0 / 79.5 |

Against the shipped arm's 8.9 / 16.7 / 68.7. The 20 s row keeps the latency the client has and
returns 30% of the CPU at better accuracy on every condition. The 25 s row returns 20% at
better accuracy too. The 30 s row returns 35% with the best clean-speech accuracy of any arm
and a two-point loss at 15 dB, at 30 s of latency. All three encode once per request and trip
no fallback.

One rule sits behind all three: the encoder's context must extend past the audio. A 20 s chunk
gets its 4 s of silence from `-ac 1200`, and a 30 s chunk cannot get any. That is why the 30 s
row takes `-nt` and the 20 s row takes both.

## Confirmation on a second speaker

The three configurations re-run on a second chapter, a different speaker whose audio the model
finds easier (baseline WER 2.8 / 11.5 / 57.9). Two rotated passes, the same unit and reference.
[HW sweep]

| | CPU (clean / 15 dB / 5 dB) | WER clean / 15 dB / 5 dB |
|---|---|---|
| 20 s chunks, `-nt -ac 1200 -sns`, ladder off | 0.714 / 0.651 / 0.708 | 2.1 / 9.9 / 56.8 |
| 25 s chunks, `-nt`, ladder off | 0.809 / 0.727 / 0.797 | 1.2 / 10.7 / 55.4 |
| 30 s chunks, `-nt`, ladder off | 0.708 / 0.619 / 0.647 | 1.9 / 16.0 / 60.4 |
| 30 s chunks, `-nt -sns`, ladder off | 0.703 / 0.615 / 0.648 | 1.9 / 15.8 / 60.3 |

The ranking holds. The 20 s and 25 s rows beat the shipped arm's WER on every condition on
both chapters. The 30 s row is the cheapest on both chapters and loses on the noisy channel on
both. It gives up 4.5 points at 15 dB here and 2.3 on the first chapter. Its cut always lands at the
window's edge with no silence after it. `-sns` changes nothing on a 30 s chunk, which is
consistent with the bracket runs being an end-of-chunk symptom.

## The deployed governor

A stock board runs a load-sampling governor and nothing pins the server. Two arms re-ran that
way, `ondemand` on every core and no `taskset`, one pass on the first chapter. [HW sweep]

| | CPU (clean / 15 dB / 5 dB) | Request wall (clean / 15 dB / 5 dB) |
|---|---|---|
| Defaults, `performance`, pinned | 1.058 / 0.856 / 0.831 | 6.5 / 5.3 / 5.1 s |
| Defaults, `ondemand`, unpinned | 1.324 / 1.048 / 1.016 | 9.1 / 7.1 / 7.0 s |
| 20 s stack, `ondemand`, unpinned | 0.832 / 0.812 / 0.694 | 5.5 / 5.4 / 4.6 s |

Under the deployed governor the defaults cost 25% more core-seconds and 40% more latency than
pinned at `performance`. The process blocks on the NPU with idle threads, the governor parks
the big cores, and the host half of every request runs slow. The stack's ratio against the
defaults holds at 0.69 either way (0.63 / 0.77 / 0.68 against the deployed defaults). So a
stock board gets the stack's 31%, and pinning at `performance` returns another 20% on top. That
is the bias the governor page measured on LLM prefill, seen from the service side.

## On a vendor kernel

The recommendations hold on the vendor kernel. The board was a second RK1 on
`6.1.172-vendor-rk35xx` and `rknpu` 0.9.8, with ggml-rocket through rknpu-submit. Every
configuration's ratio against the shipped arm lands within 0.03 of the mainline one [HW sweep, 2
passes, 2026-09-26]. The setup was the first chapter, NPU 1000 MHz, DDR pinned at 2112 MHz, CPU
governor `performance` and `taskset -c 4-7`. The model file and chunk sets were the same, and
the whisper.cpp commit was `d09f61a`:

| | CPU (clean / 15 dB / 5 dB) | Mainline RK1 | WER clean / 15 dB / 5 dB |
|---|---|---|---|
| CPU only | 1.623 / 1.706 / 1.755 | 1.645 / 1.730 / 1.813 | 8.9 / 16.8 / 70.7 |
| NPU, shipped (the reference) | 1.000 / 1.000 / 1.000 | 1.000 | 8.9 / 16.7 / 70.7 |
| 20 s, `-nt -ac 1200 -sns`, ladder off | 0.639 / 0.777 / 0.718 | 0.638 / 0.774 / 0.689 | 7.3 / 13.8 / 63.1 |
| 30 s, `-nt`, ladder off | 0.641 / 0.743 / 0.575 | 0.639 / 0.741 / 0.557 | 5.3 / 19.0 / 79.5 |

The shipped arm's request wall is 6.2 / 5.0 / 4.7 s, 4-8% under the mainline board's. Its
absolute cost is 1.022 / 0.826 / 0.768 core-seconds per audio-second.

The two ladder-off configurations reproduce the mainline WER to within half a point. The two
arms that keep the temperature ladder match it on clean speech and at 15 dB. At 5 dB, where the
ladder fires most, they read 2-4 points worse. Each board agrees with itself across both passes.
**[expected]** The ladder samples at a nonzero temperature, so a different build's arithmetic
takes a different path through it. The whisper.cpp commit and CPU flags differ between the two
boards, and neither was isolated.

On a stock vendor board, DDR runs under `dmc_ondemand`, and that slows the CPU-only arm and not
the NPU arms, see [cpu-governor-and-offload.md](cpu-governor-and-offload.md). An A/B there
needs DDR pinned as well as the CPU governor.
