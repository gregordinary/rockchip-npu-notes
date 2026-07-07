# Decode is GEMV-bound — it stays on the CPU, and that is structural

Token-by-token LLM **decode** (M=1, one query row against the whole weight matrix) is a
**GEMV**, not a GEMM. On the `rocket` path the NPU is **~82× slower at M=1** than at the
batched GEMM it was built for [HW sweep]; software pads `M==1` to 4 and the ggml backend
gates NPU matmul at `ROCKET_MIN_M=4`, so decode runs on the A76 cores. The NPU is a
**prefill / batched-GEMM / encoder** engine. This is the settled split, and the analysis
below explains why it is structural rather than a tuning gap we have not yet closed.

## Why GEMV does not benefit from the NPU

GEMV moves one weight byte per **two** FLOPs: every parameter is read from DDR exactly
once and used once. So decode throughput is set by **memory bandwidth**, not by the MAC
array — the same conclusion as prefill ([not-mac-bound.md](not-mac-bound.md)) but for a
different reason. Prefill is dispatch/DMA-floor-bound with the MACs idle; decode is
DDR-bandwidth-bound with *both* the MACs and the dispatch path idle.

The RK3588 NPU and the A76 cluster sit behind the **same LPDDR controller**. Whichever
engine runs the GEMV, the wall is the same byte stream out of DDR. The NPU adds a
per-submit dispatch/fence floor and a host cube scatter/de-tile on top of that shared
bandwidth, so for M=1 it can only lose. marty1885's RWKV port measured exactly this:
NPU GEMV 83 ms/token vs CPU 61 ms/token at M=1, K=N=1024, "GGML 0.1 ms vs RKNN 0.2 ms"
— and llama.cpp's NEON quantized GEMV is already a hand-tuned, bandwidth-saturating
kernel. There is no host-side lever that makes a fixed-function convolution pipeline beat
a SIMD GEMV at the same DDR bandwidth.

## The GEMV-optimization re-look (2026-06-29)

A re-examination of dedicated GEMV-engine work — **Hummingbird+** (Li et al., *FPGA '26*)
and the **llama.cpp ARM GEMV thread**
(ggml-org/llama.cpp#722) — confirms the split rather than opening a new lever:

- **The engine techniques do not port to fixed silicon.** Hummingbird+'s GEMV speedups —
  DSP pre-adder operand packing, `INMODE` gating, BREG/BCASCREG cascade chains, the
  double-data-rate LUT-mux elimination, the DOT/AXPY mode switch — are **FPGA datapath
  microarchitecture** (Zynq UltraScale, 140 DSPs, <1K LUTs). The RK3588 NPU is a
  **fixed-pipeline, non-programmable** CNA→CORE→DPU convolution processor; there is no
  reconfigurable DSP fabric to synthesize these into. They are reference designs for a
  *different medium* (an FPGA or an ASIC), not an action for our path.
- **The paper's own thesis is that decode is memory-bound** ("memory bandwidth … emerges
  as the primary bottleneck"), and its FPGA reaches only ~2× a DDR4 CPU *under the same
  bandwidth*. On the RK3588 the NPU has **no bandwidth advantage** over the A76 cores to
  begin with — they share the controller — so the realizable headroom over CPU GEMV here
  is below even that 2×, before the NPU's dispatch/scatter overhead.
- **Even the vendor on-NPU decode is bandwidth-bound.** Hummingbird+ cites RKNN-LLM at
  "nearly 10 token/s on a 3B LLM on RK3588" — the proprietary W4A16 path runs decode *on*
  the NPU and lands in the same band a CPU Q4 decode reaches on this chip. On-NPU decode
  is possible; it is not *faster*.

### What does transfer (model/format levers, not NPU code)

The genuinely-portable ideas in this literature reduce **bytes moved per token**, which
is the only thing that helps a bandwidth-bound decode — and they are already available in
stock llama.cpp on the CPU side:

- **Dual-precision W4 / KV8** (Hummingbird+ §"Dual Precision Operand Packing"): 4-bit
  linear weights + 8-bit KV cache. In llama.cpp this is a Q4_K (or Q4_0) GGUF plus
  `--cache-type-k q8_0 --cache-type-v q8_0`. Fewer weight/KV bytes per token ⇒ faster
  decode, directly.
- **MoE models**: an MoE that activates a small expert subset per token (the paper runs
  GPTQ-4bit Qwen3-30B-A3B — 3B active of 30B) moves only the active-expert bytes per
  step, so it decodes far faster per unit of bandwidth than a dense model of equal
  quality. This is the highest-leverage decode lever in the paper, and it is a
  **model-selection** decision: pick an MoE GGUF, let the NPU take prefill, let the CPU
  take the lean MoE decode.

The takeaway is a deployment recommendation, not a kernel: **for fast local generation on
this chip, run a 4-bit (ideally MoE) model — NPU prefill, CPU decode.** No fixed-silicon
GEMV kernel changes that.

## NPU-assisted generation, if ever pursued

The only way to put decode-class work back on the NPU is to **turn M=1 into M>1** so it
becomes the batched GEMM the NPU is good at: continuous batching across concurrent
sessions, or speculative decoding where a cheap CPU draft proposes K tokens and the NPU
**verifies all K in one M=K prefill pass**. The closest prior art is **Medusa-style**
speculative decode (lightweight multi-head draft on the frozen backbone + static
tree-attention verifier, purpose-built for static-graph accelerators; ~1.35× on short
sequences, and it confirms long sequences stay memory-bandwidth-bound). It is a multi-week
research project with a modest ceiling
and a structural tension with the static-graph execution model; recorded as the design to
copy *if* NPU-assisted generation is ever taken on, not a queued task.

## Bottom line

Decode-on-CPU is not an unfinished optimization — it is where the bandwidth math puts the
work on this chip. The GEMV-engine literature reinforces it: the speedups there belong to
reconfigurable hardware, and the portable part is a model/format choice (4-bit, MoE) that
lives entirely in stock llama.cpp. Target the NPU at prefill, the Whisper encoder, and
the SigLIP vision encoder — the batched-GEMM regimes where it wins.
