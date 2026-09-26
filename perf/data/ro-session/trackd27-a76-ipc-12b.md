# Per-symbol IPC on the A76s, and the lever ranking the instruction histogram had inverted

`gemma4-12b` F16, PINNED as the published `pin76t4` arm: `GGML_BACKEND_PATH`, `taskset 0xf0`,
`-t 4`, `-p 2048 -n 0 -r 3`, governor `performance`, NPU at 600 MHz, **streamed** (0 resident
lines in the arm's teardown). Driver `perf/data/trackd27-a76-ipc-12b.sh` (md5
`575fb2f775ec66b4d038009211af8e7c`). `libggml-rocket.so` md5
`61f02a02345f3b2e27cef45de8ade0c2`. [HW readout 2026-09-03, RK1.]

`perf record -a -e armv8_cortex_a76/cpu_cycles/ -e armv8_cortex_a76/inst_retired/ -F 299`, leaf-IP
attribution, **both events in ONE capture** so every symbol's two shares share a run. **307K cycle
samples and 246K instruction samples, 0 lost**, 397 s. The arm read **21.12 t/s** against the
published pinned mean of 21.11 and trackd24's 21.19, so the capture cost nothing measurable.

## Why one capture and not two

The deliverable is a RATIO, and taking cycles in a second run puts an unbudgeted run-to-run term
exactly on top of the effect. The A76 PMU has six programmable counters, so two events do not
multiplex; the driver asserts that with a `perf stat` before it records, and both counts come back.

**The idle trap is real and it is 1.67x.** `perf record -a` samples `swapper`, which is **3.70% of
cycles against 2.21% of instructions**. Every share below is renormalised over `llama-bench`
samples alone (93.06% of cycles, 97.03% of instructions), because a symbol whose cycle share falls
purely because the idle denominator grew would fake exactly the effect this arm looks for.

Whole-capture IPC is **1.269** and `llama-bench`-only IPC is **1.323**, against the `a76_ipc`
column's 1.265-1.275 for a streamed arm of this unit -- two instruments agreeing, which is what
licenses reading the per-symbol numbers.

## By shared object

Shares of non-idle `llama-bench` A76 cycles and instructions, with the cap each implies at this
unit's `a/phi` of **0.255**. That is a marginal exchange rate rather than a host share, and it was
measured on the residency knob, so a cap below is exact for the weight pack and imports the pack's
exchange rate for everything else.

| object | cycles | instructions | IPC | cap on removing it entirely |
|---|---:|---:|---:|---:|
| `libggml-rocket.so` | **45.29%** | 28.00% | **0.82** | 11.55% |
| `libggml-cpu.so` | 18.74% | 33.13% | 2.34 | 4.78% |
| `[kernel.kallsyms]` | 15.67% | 15.83% | 1.34 | 4.00% |
| `libc.so.6` | **9.70%** | 2.93% | **0.40** | 2.47% |
| `libm.so.6` | 8.04% | 15.02% | 2.47 | 2.05% |
| `libggml-base` | 1.33% | 2.56% | 2.54 | 0.34% |
| `libgomp` | **0.96%** | 2.28% | **3.15** | **0.24%** |

## The ranking inverts, and a measured lever says which unit is right

| term | cycles | instructions | IPC | cap |
|---|---:|---:|---:|---:|
| attention path, 6 symbols | **22.93%** | 40.68% | 2.35 | **5.85%** |
| weight pack, `mm_pack_weights` + `_seg` | **23.05%** | 4.59% | **0.26** | **5.88%** |
| output unpack, 2 symbols | 0.97% | 1.76% | 2.41 | 0.25% |

**In instructions the attention path is 8.9x the weight pack. In cycles they are the same size.**
The pack is a memory-bound scatter at IPC 0.26 and the attention path is arithmetic at IPC 2.35,
so an instruction histogram under-reads one by 5.0x and over-reads the other by 1.8x.

**A lever that is already measured settles which unit to use.** `ROCKET_F16_RESIDENT=auto` removes
the per-call weight pack for 87% of this model's weights and is worth **1.0614x**, which is
**5.79%** of the streamed wall. The cycles cap over the same two symbols is 23.05% x 0.255 x 0.87 =
**5.12%**, so the enumerated pair accounts for 88% of a gain that is on record and the remainder
is the kernel-side per-call BO work the two symbol names do not cover. The instruction cap over
the same pair is **1.02%**, which is **5.7x below a measured gain** and therefore not a cap at all.
**An instruction share is not a time share, and here that is not a caveat but a factor of five.**

**And the leftover 13% checks from the other end too.** The cycles cap over the whole pack is
5.88% and residency reaches 87% of it, so the streamed remainder is capped at **0.76%** of wall.
An earlier derivation from measured walls puts that same remainder at **0.9%** -- 42
weights of 2712 MB out of 20790. Two routes to a term that a profile bucket had priced at over a
quarter of the wall.

## `libgomp` is 0.96% of cycles, and the spin story had the sign backwards

The four addresses `0x22440`, `0x2244c`, `0x2276c` and `0x22778` carry 0.63 / 0.60 / 0.49 / 0.47 of
the instruction share and 0.34 / 0.32 / 0.10 / 0.11 of the cycle share. As a term they are
**2.26% of instructions and 0.93% of cycles, at IPC 3.20** -- the highest IPC of any aggregate in
the profile.

**A GOMP barrier spin inflates the instruction count, not the cycle count.** It is a tight,
dependency-free loop of cheap instructions, so it retires more than three per cycle. Every reading
that treated the instruction share as an upper bound expressed "in the wrong unit" was right about
the unit and wrong about the direction: converting it moves the term DOWN by 2.4x rather than up.

**So the `OMP_WAIT_POLICY` / `GOMP_SPINCOUNT` lever is closed at 0.24% of prefill wall**, and that
is a cap over the whole object rather than over the spin alone. The retired 0.7% figure was
numerically in the neighbourhood by a route that could not have known -- it read an A55 instruction
share as a time share and capped only the cluster pinning removes.

## `libc` at IPC 0.40 was invisible in the instruction histogram

`libc.so.6` is 2.93% of instructions and ranks below `libggml-base`. In cycles it is **9.70%**, the
fourth-largest object, at IPC 0.40, and its top entries are a run of unnamed addresses around
`0xa64c4`-`0xa64f8` -- the string/memory routines. That is **2.47% of prefill wall** in a term no
previous profile here had ranked at all.

## The per-symbol table

Shares renormalised over `llama-bench`; cap at `H/t` = 0.255.

| symbol | object | cycles | instructions | IPC | cap |
|---|---|---:|---:|---:|---:|
| `mm_pack_weights` | ggml-rocket | 11.82% | 1.82% | 0.20 | 3.01% |
| `mm_pack_weights_seg` | ggml-rocket | 11.23% | 2.76% | 0.33 | 2.86% |
| `expf` | libm | 7.72% | 14.13% | 2.42 | 1.97% |
| kernel `0x...57450` | kernel | 6.77% | 6.67% | 1.30 | 1.73% |
| `ggml_compute_forward_flash_attn_ext_tiled` | ggml-cpu | 6.72% | 12.17% | 2.40 | 1.71% |
| kernel `0x...574d0` | kernel | 5.91% | 7.23% | 1.62 | 1.51% |
| `ggml_compute_forward_glu` | ggml-cpu | 5.18% | 11.10% | 2.84 | 1.32% |
| `rocket_mm_batch_run` | ggml-rocket | 4.57% | 1.80% | 0.52 | 1.16% |
| `host_softmax_rows` | ggml-rocket | 3.35% | 4.10% | 1.62 | 0.85% |
| `feat_scatter_into` | ggml-rocket | 3.33% | 1.28% | 0.51 | 0.85% |
| `mm_compute_kacc` | ggml-rocket | 2.90% | 1.12% | 0.51 | 0.74% |
| `fa_mask_scores` | ggml-rocket | 2.86% | 6.76% | 3.13 | 0.73% |
| `ggml_compute_forward_rms_norm_mul_fused` | ggml-cpu | 2.61% | 5.33% | 2.70 | 0.67% |
| `ggml_backend_rocket_flash_attn` | ggml-rocket | 1.92% | 3.06% | 2.11 | 0.49% |
| `rocket_pack_activations` | ggml-rocket | 1.28% | 2.35% | 2.43 | 0.33% |
| `libgomp`, four addresses | libgomp | 0.93% | 2.26% | 3.20 | 0.24% |
| `expf@plt` | ggml-rocket | 0.37% | 0.45% | 1.64 | 0.09% |

## What a green result does not show

Leaf-IP attribution cannot say who CALLED a symbol, and the `expf@plt` entry says at least some of
`expf` is reached from `libggml-rocket`. IPC is a property of the symbol AS COMPILED HERE, so a
cap bounds what removing the current code buys and says nothing about what a vectorised
replacement would cost. A sampled profile under-reports short leaf functions in both events, and
the two events are affected differently -- a symbol shorter than one sample period biases the
cycle side and the instruction side by different amounts, which is a floor under every IPC here
that no arm in this session bounds. One model, one build, one shape, streamed. Sampling perturbs
the run, though at 21.12 t/s against 21.11 published it perturbed this one by less than the
run-to-run spread.
