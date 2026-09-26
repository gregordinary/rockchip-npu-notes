<!-- A55 instruction attribution: WHAT runs on the little cluster?   -p 2048 -n 0 -r 3
     qwen35-9b Q4_K_M, ROCKET_QUANT_RESIDENT=auto -b 2048 -ub 2048, UNPINNED.
     perf record -a -e armv8_cortex_a55/inst_retired/ -F 499, leaf-IP attribution only.
     so=61f02a02345f3b2e27cef45de8ade0c2   board: RK1, 600 MHz, governor performance

     THE QUESTION. `librocketnpu` pins every worker to a big core, so the little-cluster
     instructions the readout's `a55_inst_share` column counts belong to llama.cpp's own threads
     [source-confirmed, `rocket_affinity.c`]. Whether they are `ggml_barrier` spin or real graph
     work decides whether anything but cores can reach the term `taskset 0xf0` removes.

     WHAT THIS CANNOT SAY. A symbol histogram is a PROPORTION, not a count, so nothing here sizes
     the term in wall. Leaf-IP only, so it cannot say who CALLED a symbol. A sampled profile
     under-reports short leaf functions and a spin loop is a long leaf, so the instrument is
     biased TOWARD finding spin -- which is what makes an absent barrier a strong negative. The
     t/s below is taken under sampling and is not comparable to a campaign number.

     THE REPORT SECTIONS BELOW ARE EMPTY IN THIS CAPTURE and the reason is the harness, not the
     data: the script chowned the file to the invoking user and then read it under `sudo`, and
     `perf report` refuses a file owned by neither the current user nor root. It printed nothing
     and exited 0. The record line's "97524 samples" is what says the capture succeeded. The
     histogram was read afterwards with `perf report` as the owning user, and is transcribed
     under the log. `trackd18-a55-symbols.sh` now reads it as the owner.
-->

[2026-09-02T04:07:15+00:00] start  so=61f02a02345f3b2e27cef45de8ade0c2
--- symbol availability ---
  llama-bench: not stripped  dynsym=171  symtab=171
  libggml-rocket.so: not stripped  dynsym=1794  symtab=1794
  libggml-cpu.so: not stripped  dynsym=1918  symtab=1918
  libggml-base.so:   dynsym=2318  symtab=2318
--- perf event availability ---
 Performance counter stats for 'system wide':

           9732838      armv8_cortex_a55/inst_retired/                                        

       1.004569254 seconds time elapsed

[2026-09-02T04:07:18+00:00] warmup
[2026-09-02T04:08:33+00:00] record
[2026-09-02T04:13:34+00:00] record rc=0 wall=301s  data=6420828B
[ perf record: Woken up 21 times to write data ]
Failed to open /proc/schedstat
[ perf record: Captured and wrote 6.108 MB <data>/trackd19b-f16window-repeat/a55.data (97524 samples) ]
[rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21032MB (MemAvailable 30568MB - reserve 9535MB, no swap)
[f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         31.43 ± 0.16 |
--- perf report: comm ---
--- perf report: comm,symbol (top 45) ---
--- perf report: dso ---
[2026-09-02T04:13:35+00:00] ALL-DONE

## perf report --sort comm  (read as the owning user)
```
BY COMMAND
  97.20%  llama-bench
   1.29%  swapper
   0.85%  kworker/u32:* (three)
   0.33%  sshd

BY SYMBOL (llama-bench unless noted)
```

## perf report --sort comm,symbol  (read as the owning user)
```
BY SYMBOL (llama-bench unless noted)
  20.79%  ggml_compute_forward_gated_delta_net
  20.57%  ggml_vec_dot_f32
  17.31%  ggml_compute_forward_ssm_conv
   7.10%  ggml_compute_forward_rms_norm_mul_fused
   5.48%  ggml_vec_swiglu_f32
   3.70%  ggml_vec_silu_f32
   3.43%  tinyBLAS_Q0_ARM<block_q8_0>::gemm<3,3>
   2.58%  expf
   1.93%  ggml_compute_forward_l2_norm
   1.68%  quantize_row_q8_0
   1.48%  0x0000000000022450   <- unresolved
   1.31%  ggml_compute_forward_add_non_quantized
   1.28%  ggml_compute_forward_sigmoid
   1.26%  0x000000000002277c   <- unresolved
   1.12%  ggml_cpu_fp32_to_fp16
   0.72%  0x0000000000022440   <- unresolved
   0.71%  ggml_compute_forward_mul
   0.71%  0x0000000000022448   <- unresolved
   0.68%  ggml_compute_forward_rope_flt<float>
   0.64%  0x0000000000022774   <- unresolved
   0.61%  0x000000000002276c   <- unresolved
   0.56%  swapper [k]
   0.42%  swapper [k]
   0.34%  ggml_vec_dot_q6_K_q8_K
   0.27%  rocket_pack_activations
   0.24%  ggml_mrope_cache_init
   0.16%  rocket_unpack_output
   0.14%  ggml_vec_dot_f32@plt
   0.11%  sincosf32

READS SO FAR
```

## The six unresolved addresses

`0x22440` / `0x22448` / `0x22450` and `0x2276c` / `0x22774` / `0x2277c`, 6.4% together, fall inside
two symbols of `libggml-base.so.0.20.2`:

```
  00000000000222a0 size  660  ggml_gated_linear_attn   <- 0x22440, 0x22448, 0x22450
  0000000000022540 size  820  ggml_rwkv_wkv7           <- 0x2276c, 0x22774, 0x2277c
```

`libm.so.6`'s `__fmodl_finite` (0x221a0, 1252 bytes) covers the first three as well, so the offsets
alone do not decide it.

**The reading below is superseded, and the object is `libgomp.so.1.0.0`.** It was resolved by
model context -- this model is a hybrid gated-delta-net whose largest histogram entry is that op
family, while a long-double `fmod` has no role in an LLM prefill -- and the context was true and
selected the wrong shared object anyway. The same six addresses carry **16.02%** on `gemma4-12b`
F16, a plain transformer with neither op, where a `--sort dso` view names libgomp for exactly that
total; the `libggml-base` symbols at these offsets are graph constructors rather than kernels; and
`libggml-cpu.so` imports `GOMP_barrier`, so this is an OpenMP build whose barrier cannot appear as
a leaf symbol at all. See `trackd22-a55-symbols-12b.md`, which carries the four grounds and the
consequence for `--poll`. **The unresolved remainder is the OpenMP runtime, so the claim that no
part of the little cluster's instructions is barrier spin does not stand.**
