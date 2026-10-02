# Decode is GEMV-bound

Token-by-token LLM decode (M=1, one query row against the whole weight matrix) is a GEMV,
not a GEMM. On the `rocket` path the NPU is ~82x slower at M=1 than at the batched GEMM it
was built for [HW sweep]. Software pads `M==1` to 4 and the ggml backend gates NPU matmul at
`ROCKET_MIN_M=4`, so decode runs on the A76 cores. The NPU is a prefill, batched-GEMM and
encoder engine. This is the settled split. The analysis below explains why it is structural
rather than an open tuning gap.

## GEMV and memory bandwidth

GEMV moves one weight byte per two FLOPs: every parameter is read from DDR exactly once and
used once. So memory bandwidth sets decode throughput, and the MAC array does not. That is
the same conclusion as prefill ([not-mac-bound.md](not-mac-bound.md)), for a different
reason. Prefill is dispatch/DMA-floor-bound with the MACs idle. Decode is DDR-bandwidth-bound
with both the MACs and the dispatch path idle.

The RK3588 NPU and the A76 cluster share one LPDDR controller. Whichever engine runs the
GEMV, the wall is the same byte stream out of DDR. The NPU adds a per-submit dispatch/fence
floor and a host cube scatter/de-tile to that shared bandwidth cost, so for M=1 it can only
lose. The RWKV port by marty1885 measured exactly this: NPU GEMV 83 ms/token vs CPU
61 ms/token at M=1, K=N=1024, "GGML 0.1 ms vs RKNN 0.2 ms". The NEON quantized GEMV in
llama.cpp is already a hand-tuned, bandwidth-saturating kernel. No host-side lever makes a
fixed-function convolution pipeline beat a SIMD GEMV at the same DDR bandwidth.

## GEMV-engine literature

Dedicated GEMV-engine work confirms the split rather than opening a new lever (reviewed
2026-06-29). The sources are Hummingbird+ (Li et al., *FPGA '26*) and the llama.cpp ARM GEMV
thread (ggml-org/llama.cpp#722). They support three points:

- The engine techniques do not port to fixed silicon. Hummingbird+'s GEMV speedups
  (DSP pre-adder operand packing, `INMODE` gating, BREG/BCASCREG cascade chains, the
  double-data-rate LUT-mux elimination, the DOT/AXPY mode switch) are FPGA datapath
  microarchitecture (Zynq UltraScale, 140 DSPs, <1K LUTs). The RK3588 NPU is a
  fixed-pipeline, non-programmable CNA→CORE→DPU convolution processor, with no
  reconfigurable DSP fabric to synthesize these into. They are reference designs for a
  different medium (an FPGA or an ASIC), not an action for the `rocket` path.
- The paper's own thesis is that decode is memory-bound ("memory bandwidth … emerges as
  the primary bottleneck"). Its FPGA reaches only ~2x a DDR4 CPU *under the same
  bandwidth*. On the RK3588 the NPU has no bandwidth advantage over the A76 cores, since
  they share the controller. So the realizable headroom over CPU GEMV here is below even
  that 2x, before the NPU's dispatch/scatter overhead.
- Even the vendor on-NPU decode is bandwidth-bound. Hummingbird+ cites RKNN-LLM at
  "nearly 10 token/s on a 3B LLM on RK3588". The proprietary W4A16 path runs decode *on*
  the NPU and lands in the same band a CPU Q4 decode reaches on this chip. On-NPU decode
  is possible, and it is not faster.

### Portable model and format levers

The portable ideas in this literature are model and format levers, not NPU code. They reduce
bytes moved per token, which is the only thing that helps a bandwidth-bound decode. They are
already available in stock llama.cpp on the CPU side:

- **Dual-precision W4 / KV8** (Hummingbird+ §"Dual Precision Operand Packing"): 4-bit
  linear weights and an 8-bit KV cache. In llama.cpp this is a Q4_K (or Q4_0) GGUF plus
  `--cache-type-k q8_0 --cache-type-v q8_0`. Fewer weight/KV bytes per token give a faster
  decode, directly.
- **MoE models**: an MoE that activates a small expert subset per token moves only the
  active-expert bytes per step. The paper runs GPTQ-4bit Qwen3-30B-A3B, 3B active of 30B.
  So an MoE decodes far faster per unit of bandwidth than a dense model of equal quality.
  This is the highest-leverage decode lever in the paper, and it is a model-selection
  decision. Pick an MoE GGUF, let the NPU take prefill, and let the CPU take the lean MoE
  decode.

The takeaway is a deployment recommendation, not a kernel. For fast local generation on this
chip, run a 4-bit (ideally MoE) model, with NPU prefill and CPU decode. No fixed-silicon GEMV
kernel changes that.

## NPU-assisted generation

The only way to put decode-class work back on the NPU is to turn M=1 into M>1. The work then
becomes the batched GEMM the NPU is good at. One route is continuous batching across
concurrent sessions. The other is speculative decoding, where a cheap CPU draft proposes K
tokens and the NPU verifies all K in one M=K prefill pass.

The closest prior art is Medusa-style speculative decode. It pairs a lightweight multi-head
draft on the frozen backbone with a static tree-attention verifier, purpose-built for
static-graph accelerators. It reports ~1.35x on short sequences, and it confirms that long
sequences stay memory-bandwidth-bound. It is a multi-week research project with a modest
ceiling and a structural tension with the static-graph execution model. This note records it
as the design to copy if NPU-assisted generation is ever taken on, not as a queued task.

## Summary

Decode-on-CPU is where the bandwidth math puts the work on this chip. It is not an unfinished
optimization. The GEMV-engine literature reinforces that. The speedups there belong to
reconfigurable hardware, and the portable part is a model/format choice (4-bit, MoE) that
lives entirely in stock llama.cpp. Target the NPU at prefill, the Whisper encoder, and the
SigLIP vision encoder: the batched-GEMM regimes where it wins.
