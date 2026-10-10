<!-- Raw whisper-bench output backing perf/benchmarks.md (ASR / Whisper encoder). Whisper
     encoder MUL_MATs offloaded to the RK3588 NPU through the ggml-rocket drop-in .so;
     whisper-bench "-w 0" times the encoder in isolation (the decoder stays on the CPU on
     both backends). CPU baseline = stock whisper-bench with the backend unloaded (8-thread
     CPU); NPU = GGML_BACKEND_PATH=<libggml-rocket.so> ROCKET_KACC=1. Warm discipline: 4
     whisper-bench invocations per (model, backend), discard rep 1 (cold clock), warm mean of
     reps 2-4. RK3588 @ 600 MHz. whisper-bench -t 4. [HW sweep, 600 MHz, 2026-07-04].

     FAITHFULNESS/CONFIG note: the resident-weight cache and whisper's tensor names. whisper.cpp
     leaves its weight tensors unnamed, so ggml auto-names them "leaf_%d" by graph position, and
     the same "leaf_N" string denotes different weights across whisper's separate conv/encode/
     cross/decode graphs. ggml-rocket's resident-weight cache keys on the weight name, so before
     the fix below it served one weight's packed tiles for another's matmul -> a fast but garbage
     encode (whisper-bench times encode without checking it, so the bad config still benchmarks).
     Fixed in ggml-rocket.cpp rocket_weight_key: ggml-default leaf_/node_ names are rejected
     (empty key -> the weight streams per call instead of caching). Streaming is timing-neutral
     for a single-pass encoder (each weight is packed once regardless), so these numbers are
     unchanged whether measured with the fix (default) or the pre-fix ROCKET_NO_PREPACK=1
     workaround. llama.cpp weights carry real names ("blk.N.*") and never match, so their
     resident caching (and prefill speed) is unchanged: Llama-3.2-3B F16 NPU pp512 = 60.91 t/s
     after the fix vs 61.79 published. Numbers below are the faithful config.
-->

== Whisper encoder: whisper-bench encode-only, CPU vs FOSS-NPU, warm mean of reps 2-4 (ms) ==

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

== Whole-pipeline shape: encoder vs decoder cost (whisper-bench full timing, NPU run) ==

The encoder is what the NPU accelerates; the decoder (autoregressive, small-M GEMV) stays on the
CPU on both backends. whisper-bench's synthetic decode/batchd/prompt phases show the per-step
decoder cost, which is where large-v3-turbo differs from large-v3: same 32-layer encoder, but a
4-layer decoder instead of 32.

  large-v3        encode 17437 ms   decode 116.44 ms/run   batchd 202.94 ms/run   prompt 14.14 ms/run
  large-v3-turbo  encode 15564 ms   decode  19.13 ms/run   batchd  26.08 ms/run   prompt  3.12 ms/run

turbo's per-step decoder cost is ~6x lower, so for a real transcription the (NPU-accelerated)
encoder is a much larger fraction of the wall time -> the encoder's ~2.1x NPU speedup carries more
of the whole-pipeline win for turbo than for full large-v3.

== Faithfulness: transcript agreement, CPU vs NPU (jfk.wav, greedy, whisper-cli -np -nt) ==

  base.en         CPU: "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
  base.en         NPU: "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."   [identical]
  large-v3-turbo  CPU: "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
  large-v3-turbo  NPU: "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."   [identical]

Byte-identical CPU vs NPU (WER 0). The encoder-output cosine of the FOSS-NPU encoder vs whisper.cpp's
real base.en encoder is 0.9998 (rockchip-npu-notes/encodings/whisper-encoder.md, whisper-encoder
validation harness), the fp16-accumulation fidelity that underlies the identical transcript.

build: whisper.cpp v1.8.6-56-g84bd03a4; ggml-rocket DL against whisper's bundled ggml.

== Re-measured 2026-09-26: base.en at the 30 s and 20 s windows ==

Turing RK1, mainline `rocket` 1.3.0, 600 MHz, CPU governor `performance` on every policy,
`taskset -c 4-7`, `-t 4`, flash attention on (whisper.cpp's default; the unmasked encoder
attention stays on the CPU). whisper.cpp `eacbd82`, shared and DL-capable; ggml-rocket
`cfcc0f5` and rocket-userspace `c7e7c47`, built against it; `ROCKET_KACC=1`. Five
interleaved reps per arm, the arm order alternating, page cache dropped and memory compacted
before every arm. The 30 s window is `whisper-bench` (1500 positions). The 20 s window is
`whisper-cli -ac 1000` on `jfk.wav`, whose encoder time is fixed by the window, not the clip.

| window | CPU (ms) | NPU (ms) | speedup |
| ------ | -------: | -------: | ------: |
| 30 s   |  1513.5  |  1015.8  |  1.49x  |
| 20 s   |   883.6  |   619.0  |  1.43x  |

Per-rep encode (ms):

  30 s  CPU 1512.04 1512.09 1514.45 1514.91 1514.24   NPU 1024.60 1003.91 1013.17 1013.31 1024.21
  20 s  CPU  881.50  883.17  883.76  884.84  884.81   NPU  621.89  613.00  619.13  620.64  620.35

The 20 s transcript of `jfk.wav` is identical across all ten runs, CPU and NPU. For the 20 s
window against a whole-graph vendor compile of the same encoder, see
[encodings/whisper-encoder.md](../../encodings/whisper-encoder.md) §"In-model fused
integration".

== Same board against a whole-graph RKNN compile, 2026-09-26: base.en at the 20 s window ==

Turing RK1 on the vendor kernel, `6.1.172-vendor-rk35xx`, `rknpu` 0.9.8, CPU governor
`performance` on every policy, `taskset -c 4-7`. The NPU clock is pinned through its devfreq node
(`performance`, `max_freq` capped) and read back before every arm. The RKNN arm is
`whisper_encoder_base_20s.rknn` from `harvestsu/whisper-edge` `e318664` (rknn_model_zoo's fp16
encoder, rknn-toolkit2 2.3.0, input `[1,80,2000]`, output `[1,1000,512]`), run through
`librknnrt` 2.3.2 by a C harness that times `rknn_run` alone, median of 20 warm runs after 5.
The harnesses, scripts and raw files are in [rknn-encoder/](rknn-encoder/).
The drop-in arm is stock `whisper-cli -ac 1000 -t 4` on `jfk.wav` (whisper.cpp `d09f61a`) with
ggml-rocket `cfcc0f5` through rknpu-submit `db7c3ab` and rocket-userspace `c7e7c47`,
`ROCKET_KACC=1`. Five passes, the clock order alternating and the arm order rotating, page cache
dropped and memory compacted before every arm.

| arm | 1000 MHz (ms) | 600 MHz (ms) | 600 / 1000 |
| --- | ---: | ---: | ---: |
| RKNN, cores 0-2 | 344.25 | 364.36 | 1.058 |
| RKNN, core 0 | 399.61 | 421.40 | 1.054 |
| drop-in | 585.09 | 581.46 | 0.994 |
| CPU | 830.4 over both clocks | | |

Per pass (ms):

  RKNN 0-2  1000: 344.25 344.30 344.14 344.30 344.12   600: 364.49 364.36 364.27 364.34 364.37
  RKNN 0    1000: 399.60 399.60 399.64 399.62 399.61   600: 421.54 421.40 421.28 421.41 421.20
  drop-in   1000: 574.15 777.75 753.07 585.09 581.24   600: 578.88 579.70 581.46 586.81 615.70
  CPU       1000: 825.97 845.96 882.29 820.47 871.52   600: 821.08 817.61 879.62 828.87 831.99

Two drop-in passes at 1000 MHz read 753-778 ms. Their cause is not isolated, and the medians
carry the other three. The CPU arm's spread is the DDR governor, below.

Clock sweep of the RKNN arm, one process a cell, 10 runs (ms): cores 0-2 read 344.3 at 1000 MHz,
364.4 at 600 and 513.5 at 300. Core 0 reads 399.6, 421.4 and 584.9. `RKNN_QUERY_PERF_RUN`
agrees with the harness to within 1% in every cell.

Per-op time from `RKNN_QUERY_PERF_DETAIL`, core 0 (ms):

| op | 300 MHz | 600 MHz | 1000 MHz | 300 / 1000 | 600 / 1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `exSDPAttention` (6 calls) | 373.2 | 254.9 | 239.6 | 1.56 | 1.06 |
| `ConvAdd` (12) | 85.8 | 79.2 | 77.7 | 1.10 | 1.02 |
| `ConvExGelu` (8) | 56.4 | 37.1 | 34.9 | 1.62 | 1.06 |
| `exNorm` (13) | 26.3 | 22.8 | 22.3 | 1.18 | 1.02 |
| `Conv` (6) | 29.1 | 15.8 | 14.0 | 2.07 | 1.13 |
| everything else | 16.9 | 14.4 | 14.0 | 1.21 | 1.03 |
| total | 587.6 | 424.2 | 402.6 | 1.46 | 1.05 |

On cores 0-2 `exSDPAttention` is unchanged at 239.6 ms, `ConvAdd` falls to 29.1 and `Conv` to
6.6, and the total is 344.8. No op runs on the CPU (23 µs of the frame). The runtime reports
197,407 KB of memory traffic per frame.

Runtime version: `librknnrt` 2.3.0, the version the model was compiled with and the one Seeed
names, reads 344.32 and 344.43 ms against 2.3.2's 344.34 and 344.33, alternated at 1000 MHz on
cores 0-2. Its `exSDPAttention` is 239.6 ms.

DDR: `dmc_ondemand` against `dmc` pinned at 2112 MHz, three passes alternating, DDR sampled
every 50 ms. RKNN at 1000 MHz reads 344.05 against 343.95 ms, and at 600 MHz 364.16 against
364.08. The drop-in reads 577.69 against 574.14. The CPU arm reads 858.0 (847.8-876.3) against
817.3 (817.1-817.5). Under `dmc_ondemand` DDR sits at 2112 MHz for 165 of the RKNN arm's 168
samples, and at 528 or 1068 MHz for 44-58% of the CPU arm's.
