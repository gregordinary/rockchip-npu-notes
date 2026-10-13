# The weight repack in a quantized CPU baseline

llama.cpp's CPU backend repacks a quantized weight into an interleaved layout for its GEMM
kernels (`GGML_CPU_REPACK`, on by default). Every quantized CPU-against-NPU comparison in
these notes before 2026-09-29 ran both configurations on a llama.cpp built with the repack off.
Without the repack, the CPU configuration is 1.26-1.71x slower at pp2048. The NPU's lead over a
repacked CPU is therefore smaller than the published ratios [HW sweep 2026-09-29].

**Conditions:** an RK1 (RK3588), NPU clock 600 MHz, `rocket` 1.3.0, kernel 7.2.8, CPU governor
`performance`. The host is llama.cpp b11242 built with `GGML_CPU_REPACK=ON`, and the backend is ggml-rocket as of 2026-09-28. Prefill t/s at pp2048, with `-b 2048 -ub 2048` in every configuration.
The NPU configuration runs the MoE experts through the native-quant path, with all 4746 of
DeepSeek's experts resident.

**Not characterized:** pp512 and the default `-ub 512`, where the published quantized
configurations differ.

| Model (GGUF) | CPU, repack off | CPU, repack on | repack | NPU | NPU / CPU repack on | NPU / CPU repack off |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-V2-Lite (`Q4_K_M`) | 18.78 | 23.66 | 1.260x | 34.07 | **1.44x** | 1.81x |
| gpt-oss-20b (`MXFP4`) | 11.97 | 15.91 | 1.329x | 30.20 | **1.90x** | 2.52x |
| Phi-4 14B (`Q4_K_M`) | 3.38 | 5.45 | 1.613x | 16.67 | **3.06x** | 4.93x |
| Qwen3.6-27B (`Q4_K_M`) | 1.75 | 2.99 | 1.709x | 9.59 | **3.21x** | 5.48x |

## Cause

The NPU backend cannot use a repacked weight, because that layout lives in a buffer that is not
a host buffer.

On the A76, which has dotprod and no i8mm, the repack covers every weight type here: `Q4_K`,
`Q6_K`, `Q8_0` and `MXFP4`. It also covers the 3-D expert stacks under `MUL_MAT_ID` (ggml 0.25.3
`ggml-cpu/repack.cpp`) [source-confirmed]. It does not apply to F16, so the F16 comparisons
stand. Decode is a separate question. A repacked GEMV reads the same bytes per token, and it
measured no faster (Ministral-3-8B `Q4_K_M`, 3.30 against 3.72 t/s) [HW sweep 2026-07-17].

## Running both configurations from one build

A `GGML_CPU_REPACK=ON` build serves both configurations, because the repack is a runtime
choice. The flags are `llama-bench --repack 0|1`, and `-nr` / `--no-repack` in the common
arguments. They set `use_extra_bufts`, and with it off the CPU buffer list is the plain host
buffer that a `GGML_CPU_REPACK=OFF` build gives (llama.cpp b11242 `make_cpu_buft_list`)
[source-confirmed].

> [!CAUTION]
> The NPU configuration needs `--repack 0`, and without it the run falls back to the CPU with no
> error. With the repack on, the quantized weights are placed in the repack buffer, the backend
> declines them, and the NPU configuration runs on the CPU. Llama-3.2-3B `Q4_K_M` through the
> NPU backend offloads 780 matmuls with `-nr` and none without it [HW sweep 2026-09-28]. Without
> it, the run gives the CPU-repack perplexity exactly, 8.9742. The backend still loads, so a
> check that it registered does not detect the fallback. Read the `ROCKET profile` line or the
> residency outcome line instead.

The repack also changes the CPU configuration's results, not only its speed. The same GGUF has
perplexity 8.9807 unrepacked and 8.9742 repacked. Gemma-4-12B `Q4_K_M`'s greedy completion of
a raw prompt diverges between the two. For a faithfulness comparison against the NPU, use the
unrepacked CPU configuration: the NPU configuration's CPU-side ops run the same kernels.

## Short prompts and whole turns

A quantized prefill shorter than `ROCKET_MIN_M_QUANT` (512 rows) stays on the CPU, and under
`-nr` that CPU work runs the unrepacked GEMM. A short prompt is therefore slower in the NPU
configuration than on a repacked CPU, and a long one is faster [HW sweep 2026-07-17].

**Conditions:** Ministral-3-8B `Q4_K_M`, 8 threads, warm, RK1, NPU clock 600 MHz, kernel 7.1.
The test points predate llama.cpp b11242 and the later host-side cost reductions, so the
absolute rates are those of the earlier build.

| Prefill | CPU, repack on | NPU configuration, repack off |
|---|---:|---:|
| pp128, below `ROCKET_MIN_M_QUANT` | 10.14 t/s | 6.48 t/s |
| pp512 | not recorded | 19.21 t/s |
| pp2048 | 9.81 t/s | 19.33 t/s |

The repacked CPU prefill is 1.60x the unrepacked one at pp2048, 9.81 against 6.12 t/s. The loss
below `ROCKET_MIN_M_QUANT` still applies on later builds, because it follows from which kernel
runs the CPU work.

Over whole turns the NPU configuration is still faster or about equal, because decode dominates
a short turn and the repack does not speed decode. These are `llama-bench -pg` turns, with the
NPU configuration at `-b 2048 -ub 2048`:

| Turn (prompt, generated) | CPU, repack on | NPU configuration, repack off | NPU / CPU |
|---|---:|---:|---:|
| 2048, 256 | 6.71 t/s, 343 s | 9.98 t/s, 231 s | 1.49x |
| 512, 1024 | 3.40 t/s, 452 s | 3.88 t/s, 396 s | 1.14x |
| 128, 512 | 3.49 t/s, 183 s | 3.69 t/s, 173 s | 1.06x |

## Difference from the earlier tuning-matrix figures

The NPU configurations measure 1.03-1.21x higher than the 2026-08-29/30 tuning-matrix
measurement series at the same configuration. Most of the difference is the CPU governor, and
the llama.cpp version is neutral [HW sweep 2026-09-29]:

| Model | llama.cpp b11242 / b10558, both pinned | `performance` / `ondemand`, b11242 |
|---|---:|---:|
| DeepSeek-V2-Lite | 1.008 (1.011, 0.997, 1.016) | **1.130** (1.131, 1.122, 1.136) |
| Phi-4 14B | 1.006 (1.008, 1.005, 1.004) | **1.079** (1.087, 1.065, 1.085) |

The August rows recorded no governor, and the governor's effect on an offloading process is in
[cpu-governor-and-offload.md](cpu-governor-and-offload.md). About 1.08x remains on both models
between the `ondemand` configuration (30.34 and 15.49 t/s) and August (28.15 and 14.27). The
kernel image changed from 7.2.0 to 7.2.8 between the two, which is a candidate cause
[hypothesis]. Nothing here isolates it.

## Applicability

The NPU configuration's CPU-side work (the attention path, the MoE router, the gathers) runs
unrepacked by construction. A model whose host share is larger than these four's loses more of
its lead [expected].

## Evidence

### Measurement method

The ratios are paired within a pass, and they spread under 1% pass to pass. Each model ran
three rotated passes (two for Qwen3.6-27B) with a memory reset before every configuration. Raw
records are in [data/rebaseline-b11242/](data/rebaseline-b11242/).

### Running both configurations from one build

`bench-llm.sh`'s registration check cannot catch a missing `--repack 0`, because the backend
does load.

### Short prompts and whole turns

The decode controls from the same measurement: `Q8_0` decodes at 2.45 t/s repacked against
2.48 unrepacked, and F16, which is never repacked, at 1.41 both ways.

### Difference from the earlier tuning-matrix figures

A same-session A/B on the NPU configuration separates the causes. It ran three configurations,
rotated over three passes, all on the same ggml-rocket source.
