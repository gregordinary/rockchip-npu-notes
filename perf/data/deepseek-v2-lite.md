<!-- Raw llama-bench + llama-perplexity output backing perf/benchmarks.md (DeepSeek-V2-Lite). The
     clk=200 in each per-run header is an idle sample taken between the discarded warmup and the
     measured run; the NPU rides to 600 MHz under load (module loaded with
     rocket_npu_clk_hz=600000000). Warm medians, llama-bench -r 2 plus a discarded warmup. Q4_K_M
     ONLY: at 15.71B the F16 GGUF (~31 GB) does not fit the 31 GB board; Q4_K_M (9.65 GiB) fits with
     headroom. Quant and PPL both at -b 2048 -ub 2048. GGUF from mradermacher/DeepSeek-V2-Lite-GGUF
     (the base model deepseek-ai/DeepSeek-V2-Lite; converted 2024-05, MLA tensors in the pre-split
     attn_kv_a_mqa/attn_kv_b form, which build 9568 reconstructs -- the FLASH_ATTN op shapes are
     identical either way). arch deepseek2: MLA attention (kv_lora_rank 512, key_length 192 = 128
     nope + 64 rope, value_length 128, 16 heads) + MoE (64 routed + 2 shared experts, 6 routed
     active per token, 27 blocks, block 0 dense). ~2.4B of 15.71B params active per token.

     A COMBINED gap-finder -- it stacks the two current offload gaps in one model:
       1. MLA attention: FLASH_ATTN_EXT does NOT offload. llama.cpp builds the fused FA op for
          deepseek2 (with -fa 1 the FA column reads 1, no error), but the backend's supports_op
          rejects every one: MLA's key/value head dims are ASYMMETRIC (DK=192 != DV=128), which
          fails the handler's DK==DV==head_dim contract (it assumes standard GQA). The FA_TIMING
          probe stays silent even with -fa forced on -> zero FA ops reached the NPU -> attention
          runs entirely on the CPU. (The FA engagement diagnostic below is the evidence: no
          "ROCKET FA total" line under either -fa auto or -fa 1.)
       2. MoE routed FFN: GGML_OP_MUL_MAT_ID, not offloaded (the gpt-oss-20b gap), so the 6 active
          routed experts' gate/up/down matmuls stay on the CPU too.
     What DOES reach the NPU: the large MLA projections (q_a/q_b/kv_a/kv_b), the 2 always-on SHARED
     experts' gate/up/down, and lm_head -- all ordinary static-weight MUL_MAT. Those dense GEMMs are
     substantial (bigger than gpt-oss's GQA projections + no shared expert), so the NPU prefill win
     is modest-but-real (1.18-1.26x) and LARGER than gpt-oss's ~1.04x, even though DeepSeek loses
     BOTH attention and routed-expert offload. Unlike the instruct/reasoning models in this record,
     the absolute wikitext PPL (~8.2) is in the normal range (base model); the NPU-CPU delta is the
     faithfulness measure. See ../benchmarks.md Method. Generator: run_sweep_deepseek.sh, 2026-07-03. -->

== FA engagement diagnostic (pp2048, r1; a "ROCKET FA total" line == FA offloaded to the NPU) ==
--- -fa auto ---
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp2048 |         23.89 ± 0.00 |
--- -fa 1 (flash attention FORCED ON) ---
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |   1 |          pp2048 |         23.92 ± 0.00 |
NO "ROCKET FA total" line printed under either -fa auto or -fa 1 -> the FLASH_ATTN_EXT op is built
by llama.cpp (fa=1 column present, no error) but rejected by supports_op (MLA DK=192 != DV=128) ->
zero attention ops offloaded -> MLA attention runs on the CPU.

== DeepSeek-V2-Lite Q4_K_M  Fri Jul  3 19:40:28 UTC 2026 ==
### DeepSeek-V2-Lite Q4_K_M  [cpu]  19:41:22  clk=200 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |       8 |     2048 |           pp512 |         20.37 ± 0.01 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |       8 |     2048 |          pp1024 |         19.92 ± 0.05 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |       8 |     2048 |          pp2048 |         19.01 ± 0.02 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |       8 |     2048 |            tg64 |          7.73 ± 0.02 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |       8 |     2048 |    pp2048+tg128 |         15.25 ± 0.00 |

build: 7d2b45b4f (9568)

### DeepSeek-V2-Lite Q4_K_M  [npu]  19:58:15  clk=200 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |           pp512 |         24.04 ± 0.01 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp1024 |         24.85 ± 0.00 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp2048 |         23.87 ± 0.06 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |            tg64 |          7.77 ± 0.03 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |    pp2048+tg128 |         18.12 ± 0.03 |

build: 7d2b45b4f (9568)

Prefill CPU->NPU: pp512 1.18x / pp1024 1.25x / pp2048 1.26x. The NPU prefill is flat ~24 t/s across
the curve; the CPU baseline declines with M (20.4->19.0), so the win rises modestly. Larger than
gpt-oss's ~1.04x (the MLA projections + 2 shared experts are substantial dense MUL_MAT that offload),
far below the dense models' 3x+ (attention and the routed experts -- the bulk of the graph -- stay on
the CPU). Decode NPU ~= CPU (7.73/7.77 t/s, off-NPU, MoE ~2.4B active). The combined pp2048+tg128
point (CPU 15.25, NPU 18.12) implies the 128 tokens decoded after a 2048-tok prompt stream at ~3.7 t/s
on both backends -- about half the tg64-from-empty rate, MLA decode cost growing with the filled latent
KV cache -- so a long-prompt turn's stream is slower than tg64.

== DeepSeek-V2-Lite faithfulness (wikitext test, -c 512, 12 chunks, -b 2048 -ub 2048; same GGUF CPU vs NPU) ==
Absolute PPL (~8.2) is in the normal range (base model, not an instruct/reasoning model whose absolute
PPL is inflated); the NPU-CPU delta is the faithfulness measure. Per-run stderr +/- 0.38.
### PPL Q4_K_M [cpu]  Final estimate: PPL = 8.2444 +/- 0.37733
### PPL Q4_K_M [npu]  Final estimate: PPL = 8.2232 +/- 0.37601   (delta -0.26%)
