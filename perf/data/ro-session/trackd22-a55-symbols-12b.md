# The A55 symbol histogram on a plain-transformer model, and what it corrects

`gemma4-12b` F16, spelled as the published `unpinned` arm of the pinning intervention:
`GGML_BACKEND_PATH` and nothing else, no extra llama-bench args, `-p 2048 -n 0 -r 3`, unpinned,
governor `performance`, NPU at 600 MHz. Driver `perf/data/trackd22-a55-symbols-12b.sh`
(md5 `8c400d14f2ead7df3ad2b9eb286121ca`). `libggml-rocket.so` md5
`61f02a02345f3b2e27cef45de8ade0c2`. [HW readout 2026-09-02, RK1.]

`perf record -a -e armv8_cortex_a55/inst_retired/ -F 499` over one arm, leaf-IP attribution only.
**153932 samples, 0 lost**, 423 s of capture. The arm read **19.85 t/s** against the published
unpinned mean of 19.94, so the capture cost about half a percent and the histogram is of the arm
the matrix reports.

## By command

```
  95.44%  llama-bench
   2.21%  swapper
   1.71%  kworker/u32:* (four)
   0.21%  sshd-session + sshd-auth      <- this session's own polling, attributed and small
   0.14%  everything else
```

## By shared object

```
  76.69%  libggml-cpu.so
  16.02%  libgomp.so.1.0.0
   4.50%  [kernel.kallsyms]
   1.03%  libm.so.6
   0.96%  libggml-rocket.so
   0.29%  libggml-base.so.0.20.2
   0.20%  libc.so.6
   0.07%  [rocket]
```

## By symbol

```
  34.44%  ggml_compute_forward_glu
  14.38%  ggml_compute_forward_rms_norm_mul_fused
  12.32%  ggml_compute_forward_flash_attn_ext_tiled
   6.43%  ggml_cpu_fp32_to_fp16
   5.87%  0x0000000000022450          <- libgomp
   5.10%  ggml_compute_forward_mul
   2.95%  0x0000000000022440          <- libgomp
   2.95%  0x0000000000022448          <- libgomp
   2.13%  0x000000000002277c          <- libgomp
   1.46%  ggml_compute_forward_rope_flt<float>
   1.07%  0x000000000002276c          <- libgomp
   1.04%  0x0000000000022774          <- libgomp
   1.03%  sincosf32
   1.01%  ggml_compute_forward_rms_norm
   0.91%  ggml_compute_forward_add_non_quantized
   0.90%  swapper [k]
   0.87%  swapper [k]
   0.46%  rocket_pack_activations
   0.28%  ggml_fp32_to_fp16
   0.26%  ggml_vec_soft_max_f32
   0.20%  ggml_vec_dot_f16
   0.17%  rocket_unpack_output
   0.11%  ggml_cpu_fp16_to_fp32
   0.10%  ggml_backend_rocket_flash_attn
   0.10%  rocket_unpack_output_seg
   0.08%  feat_scatter_into
```

The driver's own host symbols total **0.91%** (`rocket_pack_activations` 0.46,
`rocket_unpack_output` 0.17, `rocket_unpack_output_seg` 0.10, `ggml_backend_rocket_flash_attn`
0.10, `feat_scatter_into` 0.08), on a model that streams every weight and therefore pays the
per-call pack on every call. Workers are pinned to the big cores [source-confirmed,
`rocket_affinity.c`], which is why streaming does not put the pack here.

## The six unresolved addresses are libgomp, not `libggml-base`

The same six addresses appeared in the `qwen35-9b` capture at **6.4%** together, and were
attributed there to `libggml-base.so`'s `ggml_gated_linear_attn` (0x222a0, 660 bytes) and
`ggml_rwkv_wkv7` (0x22540, 820 bytes) on the argument that the model is a hybrid gated-delta-net.
That attribution does not hold, on four independent grounds.

1. **The `--sort dso` view names the object**, and it is `libgomp.so.1.0.0` at 16.02%, which is the
   sum of the six rows to two decimal places. That view was not produced by the earlier run.
2. **They carry 16.02% on a model with neither op.** `gemma4-12b` is a plain transformer with no
   gated linear attention and no RWKV kernel, and the addresses are 2.5x larger here than on the
   model the attribution was built for.
3. **The `libggml-base` symbols at those offsets are graph CONSTRUCTORS**, not compute kernels.
   `ggml_gated_linear_attn` builds a tensor node once per node per graph build. The kernel is
   `ggml_compute_forward_gated_delta_net` in `libggml-cpu.so`, which is a separate symbol and was
   already the earlier histogram's largest entry at 20.79%.
4. **Nothing in libgomp's dynamic symbols covers them**, which is why they print raw:
   `omp_get_num_procs` sits at 0x21fe4 and `omp_get_wtime` at 0x22f00, and the whole region
   between is libgomp's unexported internals. The six fall in two runs of three consecutive
   8-byte-spaced addresses, which is the shape of two tight loops [hypothesis].

**Offsets agreeing across two shared objects is not attribution.** The earlier reading noted that
`libm.so.6`'s `__fmodl_finite` also covers the first three and chose between the candidates on
model context. The context was true and the choice was still wrong, because the addresses are a
property of the loaded objects and not of the model.

## This build barriers through libgomp, so `--poll` reaches nothing

`libggml-cpu.so` imports **`GOMP_barrier`**, `GOMP_parallel` and `GOMP_single_start`
[verified on the board, `nm -D`], so it is an OpenMP build. Two consequences follow from the
source [source-confirmed, `ggml/src/ggml-cpu/ggml-cpu.c`]:

- `ggml_barrier` is `#pragma omp barrier` under `GGML_USE_OPENMP`. It lowers into libgomp and
  **cannot appear as a leaf symbol at all**, so its absence from a histogram is guaranteed by
  construction and is not evidence about spinning.
- The entire polling machinery -- `threadpool->poll`, `ggml_graph_compute_poll_for_work`, the
  hybrid poll-then-sleep loop -- is inside `#ifndef GGML_USE_OPENMP` and **does not exist in this
  build**. That is why `--poll 0` measured 1.013x: the flag reaches no code.

So the wait policy that governs this barrier is libgomp's, set by `OMP_WAIT_POLICY` and
`GOMP_SPINCOUNT`, and it has both a spin budget and a sleep fallback. The recorded reading of
`ggml_barrier` as spinning "unconditionally with no poll budget and no sleep fallback" describes
the `#else` branch, which this build does not compile.

## What the histogram does and does not say

The A55 instructions are still overwhelmingly real graph work: GLU, the fused RMS norm, tiled
flash attention, the fp32-to-fp16 conversion, elementwise multiply and rope account for 74% of
them, and on a plain transformer every one of those is glue around ops the NPU took. **But 16.02%
is OpenMP runtime rather than graph work**, and the earlier capture's 6.4% was the same term
misread. "No part of the little cluster's instructions is barrier spin" does not stand.

A histogram is a proportion and cannot size a term in wall. Leaf-IP attribution cannot say who
called a symbol. The wall bound that exists is the pinning arm: pinning collapses
`a55_inst_share` from 0.141 to 0.0053-0.0064 on this model and returns 1.046x, so the whole
little-cluster term is worth 4.6% of prefill wall, and 16.02% of it is **0.7% [expected]** --
below this unit's 0.6-0.9% per-pass paired-ratio spread. Sampling perturbs the run, so the t/s
above is not a campaign number.
