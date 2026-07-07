<!-- Raw whisper-bench output backing perf/benchmarks.md (ASR / Whisper encoder). Whisper
     encoder MUL_MATs offloaded to the RK3588 NPU through the ggml-rocket drop-in .so;
     whisper-bench "-w 0" times the encoder in isolation (the decoder stays on the CPU on
     both backends). CPU baseline = stock whisper-bench with the backend unloaded (8-thread
     CPU); NPU = GGML_BACKEND_PATH=<libggml-rocket.so> ROCKET_KACC=1. Warm discipline: 4
     whisper-bench invocations per (model, backend), discard rep 1 (cold clock), warm mean of
     reps 2-4. RK3588 @ 600 MHz. whisper-bench -t 4. [HW sweep, 600 MHz, 2026-07-04].

     FAITHFULNESS/CONFIG NOTE — the resident-weight cache and whisper's tensor names. whisper.cpp
     leaves its weight tensors unnamed, so ggml auto-names them "leaf_%d" by graph position, and
     the same "leaf_N" string denotes DIFFERENT weights across whisper's separate conv/encode/
     cross/decode graphs. ggml-rocket's resident-weight cache keys on the weight name, so before
     the fix below it served one weight's packed tiles for another's matmul -> a fast but GARBAGE
     encode (whisper-bench times encode without checking it, so the bad config still benchmarks).
     Fixed in ggml-rocket.cpp rocket_weight_key: ggml-default leaf_/node_ names are rejected
     (empty key -> the weight streams per call instead of caching). Streaming is timing-neutral
     for a single-pass encoder (each weight is packed once regardless), so these numbers are
     unchanged whether measured with the fix (default) or the pre-fix ROCKET_NO_PREPACK=1
     workaround. llama.cpp weights carry real names ("blk.N.*") and never match, so their
     resident caching (and prefill speed) is unchanged: Llama-3.2-3B F16 NPU pp512 = 60.91 t/s
     after the fix vs 61.79 published. Numbers below are the faithful config.
-->

== Whisper encoder — whisper-bench encode-only, CPU vs FOSS-NPU, warm mean of reps 2-4 (ms) ==

| model            | enc d_model | enc layers | CPU (ms) | NPU (ms) | speedup |
| ---------------- | ----------: | ---------: | -------: | -------: | ------: |
| tiny.en          |         384 |          4 |   697.73 |   591.88 |  1.18x  |
| base.en          |         512 |          6 |  1579.98 |  1223.40 |  1.29x  |
| small.en         |         768 |         12 |  5711.46 |  3718.92 |  1.54x  |
| medium.en        |        1024 |         24 | 19268.18 | 10444.89 |  1.84x  |
| large-v3         |        1280 |         32 | 36399.91 | 17013.52 |  2.14x  |
| large-v3-turbo   |        1280 |         32 | 32957.15 | 15544.12 |  2.12x  |

Per-rep encode (ms): rep1 discarded (cold clock), reps 2-4 the warm mean.

  tiny.en   CPU  695.80 / 697.52 697.10 698.58     NPU  591.89 / 589.47 586.14 600.04
  base.en   CPU 1574.20 / 1579.26 1578.15 1582.52  NPU 1217.64 / 1224.96 1224.97 1220.26
  small.en  CPU 5677.04 / 5713.93 5714.78 5705.66  NPU 3632.70 / 3605.03 3760.24 3791.48
  medium.en CPU 18628.17 / 19299.80 19303.69 19201.06  NPU 10281.58 / 10408.40 10463.80 10462.46
  large-v3  CPU 35513.73 / 36163.69 36770.26 36265.79  NPU 16746.65 / 16922.92 17088.72 17028.92
  large-v3-turbo CPU 32123.66 / 32920.88 33020.67 32929.91  NPU 15595.08 / 15510.08 15558.57 15563.70

== Whole-pipeline shape — encoder vs decoder cost (whisper-bench full timing, NPU run) ==

The encoder is what the NPU accelerates; the decoder (autoregressive, small-M GEMV) stays on the
CPU on both backends. whisper-bench's synthetic decode/batchd/prompt phases show the per-step
decoder cost, which is where large-v3-turbo differs from large-v3: same 32-layer encoder, but a
4-layer decoder instead of 32.

  large-v3        encode 17437 ms   decode 116.44 ms/run   batchd 202.94 ms/run   prompt 14.14 ms/run
  large-v3-turbo  encode 15564 ms   decode  19.13 ms/run   batchd  26.08 ms/run   prompt  3.12 ms/run

turbo's per-step decoder cost is ~6x lower, so for a real transcription the (NPU-accelerated)
encoder is a much larger fraction of the wall time -> the encoder's ~2.1x NPU speedup carries more
of the whole-pipeline win for turbo than for full large-v3.

== Faithfulness — transcript agreement, CPU vs NPU (jfk.wav, greedy, whisper-cli -np -nt) ==

  base.en         CPU: "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
  base.en         NPU: "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."   [identical]
  large-v3-turbo  CPU: "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
  large-v3-turbo  NPU: "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."   [identical]

Byte-identical CPU vs NPU (WER 0). The encoder-output cosine of the FOSS-NPU encoder vs whisper.cpp's
real base.en encoder is 0.9998 (rockchip-npu-notes/encodings/whisper-encoder.md, whisper-encoder
validation harness), the fp16-accumulation fidelity that underlies the identical transcript.

build: whisper.cpp v1.8.6-56-g84bd03a4; ggml-rocket DL against whisper's bundled ggml.
