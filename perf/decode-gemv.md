# Decode (M=1) performance

Token-by-token LLM decode is a GEMV: M=1, one query row against the whole weight matrix. The
NPU's fp16 matmul throughput at M=1 is ~82x below its batched-GEMM peak: 11 GFLOPS at M=1 and
K=N=4096, against ~900 GFLOPS at 128×1024×8192. An external benchmark measured both through
the vendor RKNN matmul API (clehaxze, 2024-02-14, in [SOURCES.md](../SOURCES.md)). Decode
throughput is set by DDR bandwidth, which the NPU shares with the A76 cores. Software pads
`M==1` to 4, and the ggml backend offloads a matmul only from `ROCKET_MIN_M` rows (default 32,
never below 4), so decode runs on the A76 cores. The NPU runs the batched-GEMM workloads, such
as LLM prefill, the Whisper encoder and the SigLIP vision encoder.

**Conditions:** RK3588, `rocket` path, decode at M=1.

**Not characterized:** decode turned into M>1 work on the NPU, by continuous batching or
speculative decoding (§"Applicability").

## Cause

### GEMV and memory bandwidth

GEMV moves one weight byte per two FLOPs: every parameter is read from DDR exactly once and
used once. Memory bandwidth therefore sets decode throughput, and the MAC array does not.
Prefill is not MAC-bound either ([not-mac-bound.md](not-mac-bound.md)), for a different
reason: it is bound by DMA and dispatch, with the MACs idle. Decode is bound by DDR bandwidth,
with both the MACs and the dispatch path idle.

The RK3588 NPU and the A76 cluster share one LPDDR controller. Whichever engine runs the
GEMV, the limit is the same byte stream out of DDR. The NPU adds a per-submit dispatch and
fence overhead, and a host cube scatter and de-tile, to that shared bandwidth cost. At M=1 it
can therefore only be slower.

Martin Chang's (marty1885) RWKV port on the RK3588 shows it at two
levels [external: the "Porting LLM to RK3588" talk]. With its matmuls offloaded through RKNN, the
430M-parameter model decodes at 83 ms/token against 61 ms/token on the CPU. One M=1, K=N=1024
matmul takes 0.2 ms through RKNN and 0.1 ms in GGML.

The NEON quantized GEMV in llama.cpp is a hand-tuned, bandwidth-saturating kernel. No
host-side optimization makes a fixed-function convolution pipeline beat a SIMD GEMV at the
same DDR bandwidth.

### Weight-streaming bandwidth on `rocket`

The NPU's decode rate on `rocket` is the rate at which it streams resident weights through the
CBUF ([../matmul-as-conv.md](../matmul-as-conv.md) §"CBUF weight streaming"). At M 4, one core
streams ~5.4 GB/s, fp16 and int8 alike. Three cores driven from three processes on separate
fds reach ~16.4 GB/s at fp16 and ~15.4 GB/s at int8 [HW sweep 2026-10-21].

llama.cpp's CPU decode on the same board streams ~16-20 GB/s [host-computed]. Llama-3.2-3B
decodes at 7.88 t/s on 1.87 GiB at Q4_K_M, 5.62 t/s on 3.18 GiB at Q8_0, and 3.15 t/s on
5.98 GiB at F16, on 8 threads ([data/llama32-3b.md](data/llama32-3b.md)). At equal bytes per
weight, the NPU's stream therefore only matches the CPU's before any per-matmul host cost, and
decode stays on the CPU.

**Conditions:** RK3588 (Turing RK1), kernel 7.2.9, `rocket` 1.3.0, NPU clock 928 MHz, CPU
governor `performance`. Each NPU rate is the weight bytes over the median submit-to-fence time,
one submit and fence a run. The mainline kernel on this board exposes no DDR devfreq, so the
DDR runs at the rate the bootloader trained [live 2026-10-21].

An external userspace runs decode on the NPU with per-channel int4 and int8 weights, at an
implied ~24 GB/s over three cores [source-confirmed: `open_rknpu2_lm_infer`,
[../SOURCES.md](../SOURCES.md)]. It runs on the vendor `rknpu` driver at 1 GHz with the DDR
pinned at 2112 MHz. It also runs one task across all three cores, which `rocket` does not
offer.

### GEMV-engine literature

Published work on dedicated GEMV engines supports the same split (reviewed 2026-06-29). The
sources are Hummingbird+ (Li et al., *FPGA '26*) and the llama.cpp ARM GEMV thread
(ggml-org/llama.cpp#722). They support three points:

- The engine techniques do not port to fixed silicon. Hummingbird+'s GEMV speedups
  (DSP pre-adder operand packing, `INMODE` gating, BREG/BCASCREG cascade chains, the
  double-data-rate LUT-mux elimination, the DOT/AXPY mode switch) are FPGA datapath
  microarchitecture (Zynq UltraScale, 140 DSPs, <1K LUTs). The RK3588 NPU is a
  fixed-pipeline, non-programmable CNA→CORE→DPU convolution processor, with no
  reconfigurable DSP fabric to synthesize these into. They are reference designs for a
  different medium (an FPGA or an ASIC), and do not apply to the `rocket` path.
- The paper's own thesis is that decode is memory-bound ("memory bandwidth … emerges as
  the primary bottleneck"). Its FPGA reaches only ~2x a DDR4 CPU *under the same
  bandwidth*. On the RK3588 the NPU has no bandwidth advantage over the A76 cores, since
  they share the controller. The realizable headroom over CPU GEMV on the RK3588 is
  therefore below even that 2x, before the NPU's dispatch and scatter overhead.
- The vendor's on-NPU decode is also bandwidth-bound. Hummingbird+ cites RKNN-LLM at
  "nearly 10 token/s on a 3B LLM on RK3588". The proprietary W4A16 path runs decode *on*
  the NPU and measures in the same band a CPU Q4 decode reaches on this chip. On-NPU decode
  is possible, and it is not faster.

## Model and format options

The portable ideas in this literature are model and format choices, not NPU code. They reduce
bytes moved per token, which is the only thing that helps a bandwidth-bound decode. Stock
llama.cpp provides them on the CPU side:

- **Dual-precision W4 / KV8** (Hummingbird+ §"Dual Precision Operand Packing"): 4-bit
  linear weights and an 8-bit KV cache. In llama.cpp this is a Q4_K (or Q4_0) GGUF plus
  `--cache-type-k q8_0 --cache-type-v q8_0`. Fewer weight and KV bytes per token give a
  faster decode, directly.
- **MoE models**: an MoE that activates a small expert subset per token moves only the
  active-expert bytes per step. The paper runs GPTQ-4bit Qwen3-30B-A3B, 3B active of 30B.
  An MoE therefore decodes far faster per unit of bandwidth than a dense model of equal
  quality. It is the largest decode gain in the paper, and it is a model-selection decision.
  Pick an MoE GGUF, let the NPU take prefill, and let the CPU take the lean MoE decode.

For fast local generation on the RK3588, run a 4-bit model, ideally an MoE, with NPU prefill
and CPU decode. No fixed-silicon GEMV kernel changes that.

## Applicability

The result holds at M=1. Decode-class work returns to the NPU only as M>1 work, which is the
batched GEMM the NPU runs well. Two methods produce it. Continuous batching groups concurrent
sessions. In speculative decoding, a cheap CPU draft proposes K tokens, and the NPU verifies
all K in one M=K prefill pass.

The closest prior art is Medusa-style speculative decoding. It pairs a lightweight multi-head
draft on the frozen backbone with a static tree-attention verifier, purpose-built for
static-graph accelerators. It reports ~1.35x on short sequences, and long sequences stay
memory-bandwidth-bound. It is not implemented on the `rocket` path.

### Decode on the RK3576

Two RK3576 stacks put M=1 decode on the NPU. Neither measurement is reproduced in these notes
([../SOURCES.md](../SOURCES.md)). On mainline `rocket`, gahingwoo's `charsiu` decodes
Llama-3.2-1B at 1.39x the vendor's `librkllmrt` 1.3.0, at 594 MHz on a ROCK 4D. M=1 is no more
efficient there. At K=1024 int8, one output channel costs 0.111 us at M=1 and 0.202 us at M=80,
so a row costs 44 times as much at M=1. The gain comes from removing dispatch cost, on a host
with one A72 cluster [external].

Seeed's rknn-matmul-parallel runs Qwen3-ASR-0.6B's decoder as 196 `rknn_matmul_run` calls a
token. It costs 225 ms a token at fp16 and 146 at int8, against 43 ms for RKLLM's compiled
graph. A 60-row batch prefill costs 17 ms a token against 202 row by row [external].

## A Whisper decoder step on resident weights

A Whisper decoder step with every matmul on the NPU runs 2.6-2.8x slower than whisper.cpp's
4-thread CPU step [HW sweep 2026-10-10]. It uses about half of that step's CPU time, and
whisper.cpp on one thread recovers most of that saving on its own. Against whisper.cpp on one
thread, the NPU step saves 10-26% of the CPU, at 2-14% more wall-clock time [HW sweep
2026-10-10].

The NPU path keeps every decoder weight resident in fp16 on one canonical-tiling context, and
runs each matmul at M=4, one row padded. Q, K and V share one call, and the tied vocabulary
projection is padded to N 51872. LayerNorm, GELU and both attentions run on the host, on one
thread. The weights are random. Every step's logits match an fp32 host step at cosine
1.00000, with the same argmax.

**Conditions:** RK3588 (Turing RK1), mainline `rocket` 1.3.0, NPU clock ~816 MHz,
`ROCKET_KACC=1`, three workers, `taskset -c 4-7`, every CPU policy on `performance` with
`cpu-sleep` off. The CPU baseline is whisper.cpp v1.9.4 with OpenMP, on the multilingual F16
models and jfk.wav, greedy from `<|sot|><|en|><|transcribe|><|notimestamps|>`.

| Model | NPU step | NPU matmuls | Host part | NPU-path CPU | whisper.cpp step |
|---|---:|---:|---:|---:|---:|
| tiny | 10.3-10.4 ms | 7.3-7.4 ms | 3.0 ms | 7.7-7.8 core-ms | 3.7-3.8 ms |
| base | 21.1-22.1 ms | 12.6 ms | 8.5-9.6 ms | 16.3-17.3 core-ms | 8.1-8.3 ms |
| small | 58.1-59.8 ms | 33.1-33.3 ms | 24.7-26.4 ms | 42.2-43.9 core-ms | 22.4-22.6 ms |

The whisper.cpp step is `whisper-cli`'s decode time per run (`-bs 1 -nf`, 4 threads). The
wall-clock ratio is 2.71-2.80x at tiny, 2.60-2.67x at base and 2.57-2.68x at small. On
`ondemand` with `cpu-sleep` on, the ratio measures 2.35-2.5x, because whisper.cpp's step
gains more from pinning than the NPU path does
([cpu-governor-and-offload.md](cpu-governor-and-offload.md) §"whisper.cpp's CPU decoder").

### Weight stream at M=4

Timed alone, each distinct matmul fits a fixed cost per call plus a weight stream [HW sweep
2026-10-10]. On three workers a call costs 0.08-0.11 ms beyond its stream, and the stream runs
at 11.0-13.4 GB/s. On one worker the stream runs at 3.81-4.57 GB/s, so the three cores stream
in parallel. The vocabulary projection alone takes 4.4 ms at base: 53 MB at 12.2 GB/s.

A base step's matmuls take 12.6 ms against 7.8 ms of stream at the fitted rate. The rest is
the host path, ~0.13 ms a call over 37 calls: the input pack, the regcmd generation, cache
maintenance, the submit and the de-tile. On `ondemand` with one worker, the fixed cost halved
and the stream time tripled, and the base step measured 44.6 ms at 18.4 core-ms.

### The CPU it costs

The CPU figures come from timing whisper.cpp's decoder per call through libwhisper, replaying
the clip's greedy token sequence at each thread count. A log-softmax over the vocabulary between calls
stands in for the sampling. The replay therefore counts the CPU time an idle OpenMP team uses
between steps. The CPU per step is process CPU time over a replay [HW sweep 2026-10-10].

| Configuration | base step | base CPU | small step | small CPU |
|---|---:|---:|---:|---:|
| whisper.cpp, 4 threads | 8.7-9.0 ms | 33.3-34.5 core-ms | 22.7-23.1 ms | 88.7-90.4 core-ms |
| whisper.cpp, 2 threads | 12.0-12.7 ms | 23.6-24.9 core-ms | 33.5-34.0 ms | 66.2-67.1 core-ms |
| whisper.cpp, 1 thread | 19.5-19.8 ms | 19.4-19.8 core-ms | 56.7-57.3 ms | 56.6-57.1 core-ms |
| NPU path | 20.8-22.0 ms | 15.9-17.2 core-ms | 58.0-62.8 ms | 42.0-46.8 core-ms |

The whisper.cpp steps include the ~0.5 ms sampling stand-in, and the NPU steps do not. With
the stand-in added to the NPU path, the NPU path costs 0.84-0.90x of one thread's CPU at base,
and 0.74-0.83x at small. Its wall-clock time is 1.08-1.14x and 1.02-1.11x of one thread's.

Fewer threads reduce CPU time on their own. One thread costs 0.56-0.64x the CPU of four at
2.2-2.5x the wall-clock time, and two threads 0.69-0.76x at 1.35-1.5x.
`OMP_WAIT_POLICY=passive` saves at most 7% of the CPU at 4 threads and costs 7-15% of the
wall-clock time.

The NPU path's CPU time splits in two. At base, cross-attention takes 7.8-8.9 ms of the
8.3-9.4 ms host part, and self-attention 0.2 ms. The host cross-attention reads the fp16 K/V
at 2.1-2.4 GB/s, one key at a time. The matmul path takes the other 7.6-7.8 core-ms, of which
5.9-6.1 ms is kernel time. At small the split is 22.9-28.4 ms of cross-attention and
16.6-17.3 core-ms of matmul path.

### One-kick step estimate

A step chained into one kick keeps the stream and the host part, and drops each call's fixed
cost [expected]. The stream alone, the weight bytes at the fitted rate, is 4.2 ms at tiny,
7.8 ms at base and 25.2 ms at small. Against whisper.cpp's 4-thread step, those stream times
are 1.10-1.13x, 0.94-0.96x and 1.11-1.13x [host-computed]. Cross-attention on the NPU adds its
own K/V stream: 9.2 MB, 18.4 MB and 55.3 MB a step. At ~816 MHz a decoder step on this NPU
therefore stays slower than the CPU's 4-thread step, even with no host part and one kick
[expected].

The step reduces CPU time only against a CPU decoder that runs on fewer threads than it
could. Two terms make up its CPU time: the host cross-attention and the matmul path
(§"The CPU it costs"). Neither cross-attention on the NPU nor a one-kick matmul path is
implemented.

A whole-graph decoder on the vendor stack takes the other route. boundarybitlabs/rkwhisper
compiles each decoder step as one graph. The graph takes the token, a static 448-slot
self-attention cache, and every layer's cross-attention keys and values. Three mask inputs
update the cache by multiply and add rather than a scatter [source-confirmed: their exporter
and daemon]. Its daemon re-sends every input each step: 12 MB for tiny, 24 MB for base and
72 MB for small [host-computed from their model spec]. Most of it is cross-attention K/V,
constant within a window. None of its timings is reproduced here.

## Evidence

### Weight-streaming bandwidth on `rocket`

The NPU rates are `tests/gemv_bw_probe` at M 4, with each output scored exactly against an
integer reference before the timed runs. The three-core figure ran three processes at once, and
all three cores raised interrupts. The CPU rates are the `tg64` rows of
[data/llama32-3b.md](data/llama32-3b.md), measured 2026-07-03, multiplied by each GGUF's size.

### Throughput at M=1

The ratio is 900 / 11 = 81.8 [host-computed], from the plots of clehaxze's second RKNN matmul
benchmark post (2024-02-14). The 900 GFLOPS peak is at 128×1024×8192 with A and C in the native
layout, and the normal layout peaks at ~750 GFLOPS. The 11 GFLOPS is at M=1 and K=N=4096, where
the post measures both layouts at the same rate. Neither figure is from the `rocket` path, whose
decode measurement is the Whisper decoder step at M=4 (§"A Whisper decoder step on resident
weights").

### A Whisper decoder step on resident weights

The NPU path is `rocket-userspace` `tests/whisper_decode_step_probe.c`. Memory is reset before
each configuration. Each figure is a range over three passes with the configuration order
rotated, and each NPU figure is a median over 64 steps. The wall-clock ratios are paired within
a pass. The raw data and both measurement drivers are in
[data/whisper-decode/](data/whisper-decode/).

### Weight stream at M=4

The fit uses five shapes a model, under the pinned CPU configuration of the section's
Conditions block.

### The CPU it costs

`data/whisper-decode/wdec_cpu.c` takes the whisper.cpp figures. It encodes once, then replays
the clip's greedy token sequence at each thread count, in rotated order inside one process.
The NPU path ran in the same rotation, and each ratio to one thread is paired within a pass.
