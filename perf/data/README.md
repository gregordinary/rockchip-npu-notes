# Performance data appendix

The raw measurement records behind the performance notes, and the scripts that produced them.
Each record states its own conditions. The per-model LLM records keep theirs in an HTML comment
at the top of the file, which a rendered view hides: open the raw file to read it.

## LLM records

Common conditions apply unless a record's header states otherwise. The board is a Turing RK1
(RK3588), with the NPU clock at ~816 MHz (`rocket_npu_clk_hz=600000000` at a 0.80 V rail).
Throughputs are warm medians of `llama-bench -r 2` after a discarded warm-up. Quantized GGUFs
run at `-b 2048 -ub 2048`, and F16 at the defaults. The
`clk=` field in each per-run header is a sample of the requested rate, which the SCMI read-back
reports in place of the frequency ([../clock.md](../clock.md)). Its `clk=200` is an idle sample
taken between the warm-up and the measured run, and its `clk=600` is the ~816 MHz clock.

| Record | Model | Backs |
|---|---|---|
| [deepseek-v2-lite.md](deepseek-v2-lite.md) | DeepSeek-V2-Lite, Q4_K_M | [../benchmarks.md](../benchmarks.md) |
| [gemma4-12b.md](gemma4-12b.md) | Gemma-4-12B-it | [../benchmarks.md](../benchmarks.md) |
| [gpt-oss-20b.md](gpt-oss-20b.md) | gpt-oss-20b, MXFP4, MoE | [../benchmarks.md](../benchmarks.md) |
| [llama32-3b.md](llama32-3b.md) | Llama-3.2-3B-Instruct | [../benchmarks.md](../benchmarks.md) |
| [ministral3-3b.md](ministral3-3b.md) | Ministral-3-3B-Instruct-2512 | [../benchmarks.md](../benchmarks.md) |
| [ministral3-8b.md](ministral3-8b.md) | Ministral-3-8B-Instruct-2512 | [../benchmarks.md](../benchmarks.md) |
| [phi4-14b.md](phi4-14b.md) | Phi-4 (14B), Q8_0 and Q4_K_M | [../benchmarks.md](../benchmarks.md) |
| [phi4-mini.md](phi4-mini.md) | Phi-4-mini-instruct | [../benchmarks.md](../benchmarks.md) |
| [qwen35-08b-f16-rerun.md](qwen35-08b-f16-rerun.md) | Qwen3.5-0.8B, F16 | [../benchmarks.md](../benchmarks.md) |
| [qwen35-9b.md](qwen35-9b.md) | Qwen3.5-9B | [../benchmarks.md](../benchmarks.md) |
| [qwen36-27b.md](qwen36-27b.md) | Qwen3.6-27B, Q4_K_M | [../benchmarks.md](../benchmarks.md) |
| [smolvlm2-2.2b.md](smolvlm2-2.2b.md) | SmolVLM2-2.2B-Instruct, language model and vision tower | [../benchmarks.md](../benchmarks.md) |
| [tuning-matrix.md](tuning-matrix.md) | The default-against-tuned option comparison, one section per model file | [../../TUNING.md](../../TUNING.md) |

## Other workloads

| Record | Holds | Backs |
|---|---|---|
| [whisper-encoder.md](whisper-encoder.md) | `whisper-bench` encoder timings, CPU and NPU | [../benchmarks.md](../benchmarks.md) |
| [siglip-encoder.md](siglip-encoder.md) | The SigLIP-B/16 encoder on the vendor and mainline stacks | [../../encodings/siglip-encoder.md](../../encodings/siglip-encoder.md) |
| [asr/](asr/) | ASR in CPU core-seconds, on the vendor driver | [../asr-cpu-relief.md](../asr-cpu-relief.md) |
| [asr-short/](asr-short/) | Short-utterance ASR | [../asr-cpu-relief.md](../asr-cpu-relief.md) |
| [asr-service/](asr-service/) | `whisper-server` fed fixed-length chunks | [../asr-streaming-service.md](../asr-streaming-service.md) |
| [whisper-decode/](whisper-decode/) | The Whisper decoder step on resident weights | [../decode-gemv.md](../decode-gemv.md) |
| [rknn-encoder/](rknn-encoder/) | The same-board RKNN Whisper encoder comparison | [../../encodings/whisper-encoder.md](../../encodings/whisper-encoder.md) |
| [siglip-rknn/](siglip-rknn/) | The SigLIP-B/16 RKNN and SHARD comparison | [../../encodings/siglip-encoder.md](../../encodings/siglip-encoder.md) |
| [ro-session/](ro-session/) | Per-process readout sessions | [../per-process-readout.md](../per-process-readout.md), [tuning-matrix.md](tuning-matrix.md) |
| [dqc-session/](dqc-session/) | Quantized-prefill sessions with the dequantization cache engaged, and their idle audits | [tuning-matrix.md](tuning-matrix.md) |
| [rebaseline-b11242/](rebaseline-b11242/) | CPU baselines with and without llama.cpp's weight repack, for DeepSeek-V2-Lite, gpt-oss-20b, Phi-4 and Qwen3.6-27B | [../cpu-repack-baseline.md](../cpu-repack-baseline.md) |

## Scripts

| Script | Function |
|---|---|
| [bench-llm.sh](bench-llm.sh), [bench-llm-dryrun.sh](bench-llm-dryrun.sh) | The LLM benchmark driver, with the board reset and the rotated passes, and its dry run |
| [bench-tuning-matrix.sh](bench-tuning-matrix.sh) | The tuning-matrix driver |
| [llama-bench-pinwrap.sh](llama-bench-pinwrap.sh) | Runs `llama-bench` under a CPU affinity mask |
| [ro-join.py](ro-join.py), [ro-pagemap.py](ro-pagemap.py), [ro-selftest.sh](ro-selftest.sh), [ro-selftest-p1.sh](ro-selftest-p1.sh) | The per-process readout's join, page-map reader and self-tests |
| `trackd*.sh`, `trackd40-fa-dump-diff.py` | One driver per tuning-matrix and flash-attention measurement, named in the record it produced |
| [fa-threads-score.py](fa-threads-score.py), [interaction-refit.py](interaction-refit.py), [pair-ratio-sd.py](pair-ratio-sd.py) | Scoring and model-fitting over stored records |
| [moe-ballast-ladder.sh](moe-ballast-ladder.sh), [f16-resident-readout.sh](f16-resident-readout.sh), [qres512-9b-driver.sh](qres512-9b-driver.sh), [chain3.sh](chain3.sh), [chain4.sh](chain4.sh) | Drivers for single measurement series, cited where their results are |
