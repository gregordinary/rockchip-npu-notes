# The A76 symbol histogram, and the half of the libgomp cap that had never been measured

`gemma4-12b` F16, PINNED as the published `pin76t4` arm: `GGML_BACKEND_PATH`, `taskset 0xf0`,
`-t 4`, `-p 2048 -n 0 -r 3`, governor `performance`, NPU at 600 MHz. Driver
`perf/data/trackd24-a76-symbols-12b.sh` (md5 `f41ee75c8ceea96b145c2cee1815a35d`).
`libggml-rocket.so` md5 `61f02a02345f3b2e27cef45de8ade0c2`. [HW readout 2026-09-02, RK1.]

`perf record -a -e armv8_cortex_a76/inst_retired/ -F 499`, leaf-IP attribution only.
**368879 samples, 0 lost**, 395 s of capture. The arm read **21.19 t/s** against the published
pinned mean of 21.11, so the capture cost nothing measurable and the histogram is of the arm the
matrix reports.

## Why this arm exists

libgomp's wall cost was capped at 0.7% as `16.02% of the little cluster's 4.6% of wall`. Both
halves of that leak. The 16.02% is a share of A55 `inst_retired` read as a share of A55 TIME, and
a spin loop is exactly where an instruction share and a time share diverge. The 4.6% is the pin
gain -- the whole A55 contribution -- which pinning removes entirely, while `OMP_WAIT_POLICY`
changes barrier behavior on all eight cores. Under the recommended pinned configuration the
capped half is zero and the term is uncapped. No A76 histogram had ever been taken here.

## By command and by shared object

```
  97.21%  llama-bench                    32.08%  libggml-cpu.so
   2.10%  swapper                        27.92%  libggml-rocket.so
   0.22%  irq/69-fdab0000                17.13%  [kernel.kallsyms]
   0.22%  irq/71-fdad0000                14.83%  libm.so.6
   0.19%  irq/70-fdac0000                 3.05%  libc.so.6
   0.03%  everything else                 2.53%  libggml-base.so
                                          2.19%  libgomp.so.1.0.0
                                          0.10%  libllama.so
                                          0.07%  [rocket]
                                          0.05%  libstdc++ / libllama-bench-impl
```

## libgomp is 2.19%, and it is the SAME four addresses the A55 capture found

`0x22440`, `0x2244c`, `0x2276c` and `0x22778` carry 0.82 / 0.80 / 0.30 / 0.27 of the 2.19%. Those
are the addresses `trackd22-a55-symbols-12b.md` resolved to `libgomp.so.1.0.0` by its `--sort dso`
view after an earlier session had mis-attributed them to `libggml-base` graph constructors. **A
second cluster resolving the same region to the same object is an independent confirmation of that
correction.** libgomp exports only its public API, so the region stays unnamed in both captures.

**The term does not shrink under pinning.** As a share of the WHOLE instruction stream it is flat:
16.02% of A55 instructions at an unpinned `a55_inst_share` of 0.141 is **2.26%** of all retired
instructions, and 2.19% of A76 instructions at a pinned `a55_inst_share` of 0.007 is **2.17%**.
So the cluster-asymmetry story -- four pinned A76s are symmetric where four A55s against four A76s
are not, therefore less barrier spin -- is **not** what this measures; the share is the same either
way [HW readout 2026-09-02, RK1].

**What it does NOT say.** An instruction share is not a time share, and converting it needs a
cycles-based capture this arm did not buy. Not all of libgomp's instructions are spin --
`GOMP_parallel` and `GOMP_single_start` are real work distribution, and `OMP_WAIT_POLICY` only
removes the spin. So 2.19% is an upper bound on the reachable term expressed in the wrong unit,
and the honest reading is that the term is small on both clusters rather than that it is 0.7% of
wall.

## The driver's own host symbols are 27.92% here, against 0.91% on the A55s

`trackd22` found `librocketnpu`/`ggml-rocket` symbols totalling 0.91% of the A55 stream and
inferred that the reason is that the driver pins its workers to the big cores
[source-confirmed, `rocket_affinity.c`]. This capture measures the other side of that inference
directly: on the A76s the same code is **27.92%**, the second-largest object in the profile.

| symbol | share | object |
|---|---:|---|
| `expf` | 14.07% | libm |
| `ggml_compute_forward_flash_attn_ext_tiled` | 12.00% | ggml-cpu |
| `ggml_compute_forward_glu` | 11.17% | ggml-cpu |
| two kernel addresses (`0x…574d0`, `0x…57450`) | 12.59% | kernel |
| `fa_mask_scores` | 6.38% | ggml-rocket |
| `ggml_compute_forward_rms_norm_mul_fused` | 4.53% | ggml-cpu |
| `host_softmax_rows` | 4.02% | ggml-rocket |
| `ggml_backend_rocket_flash_attn` | 2.96% | ggml-rocket |
| `mm_pack_weights_seg` + `mm_pack_weights` | 4.96% | ggml-rocket |
| `rocket_pack_activations` | 2.43% | ggml-rocket |
| `ggml_cpu_fp32_to_fp16` + `ggml_fp32_to_fp16` | 3.58% | ggml-cpu, ggml-base |
| `rocket_mm_batch_run` | 1.70% | ggml-rocket |
| `rocket_unpack_output_seg` + `rocket_unpack_output` | 2.18% | ggml-rocket |
| `feat_scatter_into` | 1.15% | ggml-rocket |
| `mm_compute_kacc` | 0.97% | ggml-rocket |
| `libgomp` (four addresses) | 2.19% | libgomp |

**The attention path dominates.** `expf` 14.07%, `flash_attn_ext_tiled` 12.00%, `fa_mask_scores`
6.38%, `host_softmax_rows` 4.02%, `ggml_backend_rocket_flash_attn` 2.96% and `expf@plt` 0.44% are
**39.9%** of the A76 instruction stream between them, and the single largest symbol in the profile
is a scalar `expf` from libm. **That is an instruction share and not a wall share**, so it is a
lever CANDIDATE and not a cap: pricing it needs the host term's share of wall, which this arm did
not measure. The weight pack and the output unpack -- the terms residency and the micro-batch
knobs remove -- total 9.6% here, which is the same neighbourhood.

## What a green result would not show

Leaf-IP attribution cannot say who CALLED a symbol. A sampled profile under-reports short leaf
functions and a spin loop is a long leaf, so the instrument is biased TOWARD finding spin and the
2.19% must be read against that bias. Sampling perturbs the run. One model, one build, one shape.
