<!-- Raw llama-bench + llama-mtmd-cli + llama-perplexity output backing perf/benchmarks.md
     (SmolVLM2-2.2B-Instruct, vision-language / mtmd). GGUFs from
     ggml-org/SmolVLM2-2.2B-Instruct-GGUF: the language model (SmolLM2-class, arch `llama`,
     1.81 B; F16 / Q8_0 / Q4_K_M) plus the SigLIP-SO400M vision tower as an mmproj GGUF
     (mmproj-...-f16.gguf). "2.2B" is the combined vision+LLM param count; the LLM alone is
     1.81 B (llama-bench shows "llama ?B ... 1.81 B").

     TWO backends, TWO measurements:
       - LLM prefill/decode: stock llama-bench, the language GGUF alone, exactly as every other
         LLM in this record. CPU baseline = backend unloaded (8 threads); NPU =
         GGML_BACKEND_PATH=<libggml-rocket.so> ROCKET_KACC=1. F16 at defaults; quants at
         -b 2048 -ub 2048. Warm medians, -r 2 + a discarded warmup.
       - Vision encoder: the SigLIP tower run through llama.cpp's mtmd/clip.cpp. clip builds a
         ggml_backend_sched over [selected backend, CPU]; the NPU is attached with
         MTMD_BACKEND_DEVICE=ROCKET. clip's auto-path only probes GPU/IGPU *device types*, and
         rocket registers as an ACCEL device named "ROCKET", so the env selector is REQUIRED --
         without it the vision encoder silently runs on the CPU. mmproj is fp16 so the
         offloadable vision matmuls run fp16 on the NPU. Timing via an env-gated in-process
         repeat loop around clip_image_batch_encode's sched_graph_compute (a measurement-only
         patch on the reference llama.cpp checkout, not in our repos): CLIP_ENC_REPEAT=N,
         rep 0 = cold, reps 1..N-1 warm. performance CPU governor.

     RK3588 @ 600 MHz (module loaded rocket_npu_clk_hz=600000000). build 7d2b45b4f (9568).
     [HW sweep, 600 MHz, 2026-07-04]. See ../benchmarks.md Method. -->

== SmolVLM2-2.2B vision encoder (SigLIP-SO400M) — clip encode, CPU vs NPU, per-rep ms ==

SigLIP-SO400M vision tower: hidden 1152, intermediate 4304, 27 layers, 16 heads (head_dim 72),
patch 14, 729 tokens/image. Single image = one 729-token encode graph (no multi-tile split here).

Graph placement (ggml_backend_sched reserve):
  CPU:         1 split,  859 nodes
  NPU ROCKET: 272 splits, 913 nodes

The vision graph shatters into 272 CPU<->NPU handoffs. ggml-rocket offloads only STATIC-WEIGHT
GEMMs that clear K%32==0, N%16==0, K>=64, N>=64 (src0 == a model-parameter leaf). Per SigLIP
layer that admits exactly q/k/v/o projections + fc1 (5 GEMMs). Excluded:

  - fc2 down-proj: K=4304, 4304%32=16 -> fails K%32.
  - attention QK^T and score*V: src0 is a COMPUTED tensor (not a weight leaf), so the handler
    skips them by design -- independent of head_dim 72 also missing %32/%16.
  - every norm / softmax / GELU / residual-add / patch-embed op -- op types ggml-rocket does
    not implement.

27 layers x 5 offloaded GEMMs = ~135 NPU ops, each bracketed by CPU-only ops -> 272 splits.

  backend      cold(rep0)   warm reps (ms)                          warm median   speedup
  NPU ROCKET     8122.0     6658.6 6715.1 6686.9 6446.3 6476.3      ~6659         1.19x
  CPU            7912.1     7900.7 7946.9 7908.9 7898.2 7881.7      ~7901         1.00x

Warm: NPU 6.66 s vs CPU 7.90 s -> 1.19x. Cold single-encode (parked clock): NPU 8.12 s vs
CPU 7.91 s (~1.03x SLOWER) -- the cold NPU clock penalty inverts the small warm win, so warm
discipline is load-bearing here. tile384.jpg (384x384) and the full test.jpg give the same
729-token single-graph encode and the same timing.

== SmolVLM2-2.2B LLM (arch llama, 1.81 B) — llama-bench, CPU vs NPU, warm ==

### SmolVLM2-2.2B F16  [cpu]
| model              |   size |  params | backend | threads |         test |         t/s |
| ------------------ | -----: | ------: | ------- | ------: | -----------: | ----------: |
| llama ?B F16       | 3.38 GiB | 1.81 B | CPU     |       8 |        pp512 | 27.97 ± 0.30 |
| llama ?B F16       | 3.38 GiB | 1.81 B | CPU     |       8 |       pp1024 | 29.46 ± 0.08 |
| llama ?B F16       | 3.38 GiB | 1.81 B | CPU     |       8 |       pp2048 | 27.45 ± 0.13 |
| llama ?B F16       | 3.38 GiB | 1.81 B | CPU     |       8 |         tg64 |  5.11 ± 0.20 |
| llama ?B F16       | 3.38 GiB | 1.81 B | CPU     |       8 | pp2048+tg128 | 18.95 ± 0.02 |

### SmolVLM2-2.2B F16  [npu]
| model              |   size |  params | backend | ngl |         test |         t/s |
| ------------------ | -----: | ------: | ------- | --: | -----------: | ----------: |
| llama ?B F16       | 3.38 GiB | 1.81 B | ROCKET  |  -1 |        pp512 | 98.98 ± 1.48 |
| llama ?B F16       | 3.38 GiB | 1.81 B | ROCKET  |  -1 |       pp1024 | 71.99 ± 0.62 |
| llama ?B F16       | 3.38 GiB | 1.81 B | ROCKET  |  -1 |       pp2048 | 52.59 ± 0.02 |
| llama ?B F16       | 3.38 GiB | 1.81 B | ROCKET  |  -1 |         tg64 |  5.19 ± 0.19 |
| llama ?B F16       | 3.38 GiB | 1.81 B | ROCKET  |  -1 | pp2048+tg128 | 27.06 ± 0.07 |

### SmolVLM2-2.2B Q8_0  [cpu]  (-b 2048 -ub 2048)
| model              |   size |  params | backend | threads | n_ubatch |         test |         t/s |
| ------------------ | -----: | ------: | ------- | ------: | -------: | -----------: | ----------: |
| llama ?B Q8_0      | 1.79 GiB | 1.81 B | CPU     |       8 |     2048 |        pp512 | 26.78 ± 0.53 |
| llama ?B Q8_0      | 1.79 GiB | 1.81 B | CPU     |       8 |     2048 |       pp1024 | 27.29 ± 0.02 |
| llama ?B Q8_0      | 1.79 GiB | 1.81 B | CPU     |       8 |     2048 |       pp2048 | 23.81 ± 0.25 |
| llama ?B Q8_0      | 1.79 GiB | 1.81 B | CPU     |       8 |     2048 |         tg64 |  9.60 ± 0.61 |
| llama ?B Q8_0      | 1.79 GiB | 1.81 B | CPU     |       8 |     2048 | pp2048+tg128 | 18.50 ± 0.03 |

### SmolVLM2-2.2B Q8_0  [npu]  (-b 2048 -ub 2048)
| model              |   size |  params | backend | ngl | n_ubatch |         test |         t/s |
| ------------------ | -----: | ------: | ------- | --: | -------: | -----------: | ----------: |
| llama ?B Q8_0      | 1.79 GiB | 1.81 B | ROCKET  |  -1 |     2048 |        pp512 | 47.93 ± 0.53 |
| llama ?B Q8_0      | 1.79 GiB | 1.81 B | ROCKET  |  -1 |     2048 |       pp1024 | 47.11 ± 0.01 |
| llama ?B Q8_0      | 1.79 GiB | 1.81 B | ROCKET  |  -1 |     2048 |       pp2048 | 39.90 ± 0.01 |
| llama ?B Q8_0      | 1.79 GiB | 1.81 B | ROCKET  |  -1 |     2048 |         tg64 |  9.83 ± 0.49 |
| llama ?B Q8_0      | 1.79 GiB | 1.81 B | ROCKET  |  -1 |     2048 | pp2048+tg128 | 26.11 ± 0.05 |

### SmolVLM2-2.2B Q4_K_M  [cpu]  (-b 2048 -ub 2048)
| model              |   size |  params | backend | threads | n_ubatch |         test |          t/s |
| ------------------ | -----: | ------: | ------- | ------: | -------: | -----------: | -----------: |
| llama ?B Q4_K_M    | 1.03 GiB | 1.81 B | CPU     |       8 |     2048 |        pp512 |  27.01 ± 0.38 |
| llama ?B Q4_K_M    | 1.03 GiB | 1.81 B | CPU     |       8 |     2048 |       pp1024 |  26.25 ± 0.04 |
| llama ?B Q4_K_M    | 1.03 GiB | 1.81 B | CPU     |       8 |     2048 |       pp2048 |  24.61 ± 0.06 |
| llama ?B Q4_K_M    | 1.03 GiB | 1.81 B | CPU     |       8 |     2048 |         tg64 |  14.66 ± 3.33 |
| llama ?B Q4_K_M    | 1.03 GiB | 1.81 B | CPU     |       8 |     2048 | pp2048+tg128 |  19.48 ± 0.09 |

### SmolVLM2-2.2B Q4_K_M  [npu]  (-b 2048 -ub 2048)
| model              |   size |  params | backend | ngl | n_ubatch |         test |          t/s |
| ------------------ | -----: | ------: | ------- | --: | -------: | -----------: | -----------: |
| llama ?B Q4_K_M    | 1.03 GiB | 1.81 B | ROCKET  |  -1 |     2048 |        pp512 |  48.57 ± 0.02 |
| llama ?B Q4_K_M    | 1.03 GiB | 1.81 B | ROCKET  |  -1 |     2048 |       pp1024 |  47.43 ± 0.07 |
| llama ?B Q4_K_M    | 1.03 GiB | 1.81 B | ROCKET  |  -1 |     2048 |       pp2048 |  40.56 ± 0.07 |
| llama ?B Q4_K_M    | 1.03 GiB | 1.81 B | ROCKET  |  -1 |     2048 |         tg64 |  14.54 ± 3.26 |
| llama ?B Q4_K_M    | 1.03 GiB | 1.81 B | ROCKET  |  -1 |     2048 | pp2048+tg128 |  27.68 ± 0.18 |

(tg64 Q4_K_M variance ±3.3 t/s is real run-to-run jitter on this small model at the fast
decode rate; NPU 14.54 vs CPU 14.66 is within it -- decode is CPU-bound either way.)

== Faithfulness: LLM differential perplexity (wikitext test, 12 chunks, -b 2048 -ub 2048) ==

  model     CPU PPL              NPU PPL              delta
  F16       12.1622 +/- 0.60542  12.1540 +/- 0.60507  -0.07%
  Q4_K_M    12.7622 +/- 0.63139  12.6711 +/- 0.62578  -0.71%

Both deltas are far inside the ~+/-0.60-0.63 per-chunk stderr (the F16 NPU-CPU delta is ~75x
smaller than its stderr). Absolute PPL ~12.2 is inflated -- SmolVLM2's LLM is instruction-tuned
and models raw wikitext poorly -- so only the NPU-CPU delta is informative. NPU-faithful.

== Faithfulness: vision encoder — projected image-embedding cosine (CPU vs NPU) ==

The 81 projected image tokens (729 patches / scale_factor^2=9; 2048-dim = the LLM's embedding
width) that clip hands to the language model, dumped for vision=CPU and vision=NPU (same fp16
mmproj, greedy), compared:

  global cosine            0.99998432
  per-token mean cosine    0.99998462   (min 0.99967)
  max abs diff 0.3265,  relative L2 5.61e-03

The NPU vision encoder is faithful at cosine 0.99998 -- matching the SigLIP pillar (0.999998,
native driver) and Whisper (0.9998) bar. The residual ~0.5% relative-L2 is the fp16-accumulation
difference of the offloaded projection/fc1 GEMMs.

== Faithfulness: caption agreement (vision CPU vs NPU, Q4_K_M LLM on CPU, greedy) ==

vision=CPU: "The image is a newspaper front page from 'The New York Times,' dated March 23,
  1965. The headline, 'MEN WALK ON MOON,' is prominently displayed ... 'Astronauts Land on
  Plain; Collect Rocks, Plant Flag' and 'Voices From Moon.'"
vision=NPU: "The image is a newspaper clipping from 'The New York Times' dated March 21, 1965,
  with the headline 'MEN WALK ON MOON: ASTRONAUTS LAND ON PLANET; COLLECT ROCKS, PLANT FLAG.'
  The article is about the Apollo 11 mission, where astronauts Neil Armstrong and Edwin ..."

Both captions correctly read the same NYT moon-landing front page (same headline + "Collect
Rocks, Plant Flag" subhead) but diverge in wording. This is NOT a contradiction of the 0.99998
encoder cosine: greedy autoregressive decode amplifies the ~0.5% embedding perturbation into a
different-but-equivalent token path (the hallucinated date differs in both from the true 1969).
For a vision->LLM pipeline the embedding cosine, not the greedy caption tokens, is the encoder
faithfulness metric -- exactly as WER/cosine, not token-identity, is the metric for Whisper.

== Config / method notes ==

- Vision device selector: MTMD_BACKEND_DEVICE=ROCKET (clip's auto-path probes GPU/IGPU *device
  types* only; rocket is an ACCEL device named "ROCKET" in the deployed build-dl .so, so the
  selector is REQUIRED -- without it clip silently runs the vision encoder on the CPU). Note the
  device NAME is "ROCKET"; the current ggml-rocket source renames the device to "RK3588 NPU
  (mainline rocket driver)", so after a build-dl rebuild the selector value tracks that string.
- Measurement-only patches on the RK1 reference llama.cpp checkout (tools/mtmd/clip.cpp, NOT our
  repos): an env-gated wall-clock timer + in-process repeat loop (CLIP_ENC_TIMING /
  CLIP_ENC_REPEAT) around clip_image_batch_encode's sched_graph_compute, and a full-embedding
  binary dump (MTMD_EMB_DUMP) in the MTMD_DEBUG_EMBEDDINGS block. All default-off; normal use
  unchanged.
