# Whisper encoder on the NPU — EXP / softmax / LogSoftmax / cross-entropy / LayerNorm / conv1d / MHA / encoder block

Everything below is HW-validated on the RK1 at 600 MHz against an fp64 oracle
(gates `exp_lut_rocket`, `softmax_rocket`, `layernorm_rocket`, `conv1d_rocket`, `mha_rocket`,
`encoder_block_rocket`). The pieces compose the full Whisper/transformer **encoder block**.

## EXP LUT (`ROCKET_ACTIVATION_EXP`, enum 10)
Shifted single-table path (same as sqrt/rsqrt/reciprocal: `act_shifted_domain` + `build_lut_shifted`),
default domain `[-16,0]` (softmax: input ≤0 after row-max, output (0,1], `out_lo=0`, `S=1`). Whole
domain on the positive LUT index half ⇒ no LE/LO sign-mux glitch ⇒ **works standalone** (unlike
the build_lut_affine GELU). exp's relative interp error is ~constant `Δ²/8` (~1e-4 over 512 cells).

**HW FACT — a q=0 LUT table entry mis-decodes to ~4.0 (garbage), not 0.** exp's deep tail
(`exp(x)<1.5e-5`, x<~-11) quantizes to q=0 and read ~4, blowing up the softmax sum. Fix = floor
every shifted-table entry to **q≥1** (`build_lut_shifted`, `ROCKET_LUT_QFLOOR` default 1) — the
floored value decodes to ~3e-5 (~0 on readback), correct. sqrt/rsqrt/reciprocal never hit q=0.

## Softmax (`rocket_softmax_fp16`, src/rocket_softmax.c)
Row-wise over the last axis. Composition: **host** row-max + subtract (→ EXP's ≤0 domain; the
row-max is mandatory and matmul/the feature-reduce can only sum, never max — the only on-NPU max
is the PPU max-pool, the resident-fusion path) → **NPU** `exp` (LUT) → **NPU** row-sum (feature-axis
ones-matmul reduce, fp32) → **host** `1/s` (O(M), exact) → **NPU** per-row scale (`scale_rows`=ew_mul).
Validated to T=1500 (Whisper seq): rows sum to 1±0.0005, max_abs ~1e-4.

## LogSoftmax (`rocket_logsoftmax_fp16`, src/rocket_softmax.c)
`out = x − logsumexp(x)` — the classification / NLL-loss head and the LM log-prob output. Shares
softmax's steps 1–3 (host row-max+subtract → NPU `exp` → NPU row-sum `s`), then the tail is a
**subtract** of a per-row scalar instead of a divide: **host** `ls = log(s)` (O(M), exact — `s≥1`
since the row-max term contributes `exp(0)=1`, so `ls≥0` and `out≤0`) → **NPU** per-row broadcast
`ew_sub` (`out = (x−rowmax) − ls`). **The per-row log stays on the host, not the LOG LUT** — it is M
scalars already host-side as fp32 after the reduce read-back, so host `log` is exact and adds no
round-trip, exactly as softmax keeps `1/s` on the host (and RMSNorm keeps `rsqrt`). The DPU
[`ROCKET_ACTIVATION_LOG`](dpu-lut-activation.md) LUT is for a *large* tensor of logs, a different
use. LogSoftmax is all-additive ⇒ **better-conditioned than softmax for a gate** (no tiny-probability
relative blow-up) → check absolute error. HW: `max_abs ≤0.031` (the wide-spread amp=20 case; ~0.008
typical = fp16 storage), `Σ_n exp(out)=1`, validated to T=1500.

## Stable cross-entropy (`rocket_cross_entropy_fp16`, src/rocket_softmax.c)
`CE[m] = logsumexp(logits[m]) − logits[m][target[m]] = −logsoftmax(logits[m])[target[m]]` — the
softmax-classifier / LM NLL loss, one scalar per row, in its numerically-stable form (**never
materializes softmax** — no divide). Reuses the LogSoftmax front half (the logsumexp reduction):
**host** row-max + subtract → **NPU** `exp` (LUT) → **NPU** fp32 row-sum `s` → **host** `lse = rowmax
+ log(s)` → **host gather** `logits[m][target[m]]` → `CE[m] = lse − gathered`. So cross-entropy is the
on-NPU logsumexp + a host gather/subtract; it is strictly **cheaper than LogSoftmax** (skips the tail
per-row `ew_sub` — only the M target log-probs are needed, not the full `[M,N]` output).

**NPU FACT — there is no hardware gather on the RK3588 NPU.** No indexed/scatter read path exists
(the same fact the [reduce](feature-reduce.md) / attention notes record), so `logits[m][target[m]]`
is M scalar host index-lookups — correct and free, exactly like softmax's host `1/s` and LogSoftmax's
host `log(s)`. A gather is not a contraction or a pool, so neither the matmul nor the PPU supplies it.

**NPU FACT — CE is fp32-grade, more accurate than LogSoftmax (HW `max_abs ≤ 1.5e-4` vs 0.031).** The
loss is `lse − gathered`, both fp32/host values; the only NPU-introduced error is in `s` (the fp32 sum
of the LUT-`exp`), and `log(s)` is taken in fp64 on the host, so the error is just `log(s_npu/s_ref) ~
1e-4`. **The loss never round-trips through fp16 output storage** — that is precisely the term that
dominates LogSoftmax's error (the fp16 storage of the very-negative log-probs), and CE skips it.
`s ≥ 1` ⇒ `log(s) ≥ 0` ⇒ `CE ≥ 0` (the loss invariant, gate-checked, with one `abs_tol` of LUT slack
at the argmax row where `CE = log(s) ≈ 0`). Gate `tests/cross_entropy_rocket.c`: vs an fp64 CE oracle
across the M-tile boundary, N%32≠0, T=1500 vocab, random + forced-argmax targets, wide spread; the
host self-check `CE == −logsoftmax_ref[target]` cross-checks against the independent LogSoftmax path.
Optional follow-on: a `softmax_cross_entropy` returning the softmax too (backward grad `softmax −
onehot`) — keep the loss path stable (don't divide).

## LayerNorm (`rocket_layernorm_fp16`, src/rocket_norm.c)
`(x-mean)/sqrt(var+eps)*gamma + beta` per row. **Both reductions share one feature-reduce job by
stacking rows**: A = `[x ; x⊙x]` (2M rows) under the ones weight → first M outputs = `sum(x)`, next
M = `sum(x²)`. (A second ones-column does not give `sum(x²)` — N-columns are weighted sums of the
same A, i.e. of x, not x²; stack rows.) Host O(M) tail (mean,var,rsqrt); the affine folds to
`out = x⊙A + B`, A=r·gamma, B=beta−mean·r·gamma (one ew_mul + one ew_add). fp16-square overflow
prescale matches RMSNorm (k=ceil(log2(|x|max/223)), recover ·4^k). beta may be NULL.

## conv1d front-end (`rocket_conv1d_fp16`, src/rocket_conv.c)
Whisper's two front-end convs (KW=3, pad=1; conv1 IC=n_mels stride 1, conv2 stride 2) = a width-only
1D conv, lowered onto the validated `rocket_conv2d_fp16`. **Lower with time on the HEIGHT axis (IW=1),
not width** — the natural width-on-time layout (IH=1) overflows the feature banks. The conv tiler
tiles output rows (OH) first, so a long/many-channel sequence shrinks the per-tile height to fit one
CBUF pass. The width-on-time layout leaves OH=1 untileable and **overflows the feature banks for
Whisper IC=80/512 (gen returns -1)**. Layouts are byte-identical either way (no repack). Small
reductions bit-exact; large IC·KW within fp16-accum tolerance (the CNA accum order differs from the
host loop — 1-ULP store diffs, benign).

## Multi-head self-attention (`rocket_mha_self_fp16`, src/rocket_attn.c)
The pure attention sublayer (no LN/residual). q/k/v = x·W^T (+bias) [NPU matmul]; per head scores =
scale·(q_h·k_h^T) [NPU], softmax row-wise [NPU], ctx_h = P·v_h [NPU]; out = concat·Wo^T+bo [NPU].
Host glue: head slicing, the d_head^-0.5 scale, bias adds, the per-head V transpose (matmul needs B
as [N,K], so ctx = matmul(P, v_h^T) = P·v_h).

**Key-count alignment.** `rocket_matmul_fp16` rejects unaligned N/K (e.g. N=100). The scores matmul
has N = key count T, and ctx has K = T — so **pad the key count to Tn=(T+31)&~31 and mask the pad
score columns to −30000 before softmax** (zero pad keys give score 0, which softmax would weight as
exp(0); masking → exp underflow → ~0; the zero pad V rows add nothing to ctx). Query rows pad to
Tp=(T+3)&~3 (M%4). This makes attention correct for any T (Whisper T=1500 is itself unaligned).
cos=1.000000 vs the fp64 oracle at Whisper-base d=512/8-head incl. T%16≠0.

## Encoder block (`rocket_encoder_block_fp16`, src/rocket_encoder.c)
Whisper pre-norm: `x = x + MHA(LN1(x)); x = x + MLP(LN2(x))`, MLP = `GELU(h·Wf1^T+bf1)·Wf2^T+bf2`.
**Fully on the NPU**: both LayerNorms, all attention matmuls + every softmax, both residual adds, the
two MLP projection matmuls, and the MLP's GELU. cos=1.000000 vs the fp64 block oracle (d=256/512,
incl. T%16≠0). max_abs ~2e-3 = fp16 quant at output magnitude ~2.4.

## On-NPU GELU — the 2-pass `x·Φ(x)` (single-pass fusion fails)
GELU runs the **2-pass** route (like SiLU): `GELU(x)=x·Φ(x)`, Φ = the Gaussian CDF, a monotone `[0,1]`
function on the clean unit-LUT geometry (no QUIRK-1 flat-region spike), then a DPU EW-mul by x. In the
encoder MLP: `Φ(f1)` on the DPU LUT (`ROCKET_ACTIVATION_GELU_GATE`) → `f1·Φ(f1)` on the DPU EW-mul.
HW-validated cos=1.000000 vs true erf-GELU over `[-12,12]` (`tests/gelu_rocket.c`). **Negative result:
a fused single-pass matmul→GELU (1×1-conv-act epilogue) fails for wide FFN inputs — cos≈0.04,
max_abs=128 — the QUIRK-1 flat-tail mux spike hits en masse (fc1 output spans `GELU(x)≈0`, x≲-3). So
single-pass LUT fusion is curved-region-only; the durable on-NPU GELU/SiLU is the 2-pass `x·gate(x)`.**

## Validation against whisper.cpp (real base.en, real audio) [HW sweep]
The composition above is validated not only against the self-contained fp64 block oracle but
against whisper.cpp's REAL encoder on real audio (jfk.wav), base.en (d=512, 8 heads, d_ff=2048,
6 blocks, T=1500). Method: dump whisper's encoder input (`embd_conv` + positional embedding) and
final output (`embd_enc`, post-LN) from a stock-CPU run; feed the same input and the real model
weights to `rocket_encoder_block_fp16` on the NPU; compare per block and end to end. A
double-precision reference of whisper's EXACT math (biased-variance LayerNorm eps=1e-5, tanh-approx
GELU, attention scale 1/sqrt(d_head), K has no bias) sits between as the golden, itself confirmed
== whisper (`embd_enc` cos 0.99983).

- **Per-block, fed the golden input: cos 1.000000** (isolated — the block computation is faithful).
- **Chained through all 6 blocks: cos ≥ 0.99997**, and the **final post-LN encoder output is
  cos 0.99981 vs whisper's `embd_enc`** — far past the ≥0.99 bar (SHARD, a SOTA RK3588 VLM, runs at
  0.95).
- The **erf-vs-tanh GELU gap is a non-issue**: raw |tanh-GELU − erf-GELU| peaks at 4.7e-4 (x≈2.7),
  and at the block level the two are cos 1.0 / max_abs ~1.7e-4 (within fp16 noise). The block's erf
  GELU needs no tanh-approx variant to match whisper.

### NPU FACT — Whisper activation outliers explode the residual stream; the LayerNorm fp16-square prescale is load-bearing [HW sweep]
On real audio the encoder develops a few stable, dominant outlier channels (base.en: channel 270
reaches |x|≈1221 at every block; 67/33/64/73/509 reach ~90–200), and the residual-stream RMS jumps
from ~0.9 (block 3) to ~8.4 (block 4) as most channels grow large. Consequences:
- **fp16 x² overflows**: channel 270 squared ≈ 1.5e6 > fp16 max (65504), so a naive fp16 LayerNorm
  variance sum would be Inf. `rocket_layernorm_fp16`'s power-of-2 prescale (|x|>~223 → ·4^-k, fp32
  reduce, recover) is what keeps the encoder finite and correct here — it fires on every real
  Whisper block, not as a corner case.
- **outliers amplify fp16 chaining error**: a tiny fp16 difference on an outlier channel is
  amplified by the next block's high-gain weights, so the *intermediate* rocket-vs-whisper max_abs
  grows to ~48 by block 5 (RMS ~8.6) while cosine stays ≥0.99997. Post-LN re-normalises the
  magnitude away, so the encoder *output* is faithful (cos 0.9998); the single large surviving
  element post-LN (max_abs ~6.7) is the outlier channel scaled by its LN gamma.
- **implication**: the residual stream is the precision floor, not the matmul. Tightening past
  cos 0.9998 (if ever needed) means carrying the residual/FFN in higher precision (the fp32-output
  matmul `rocket_matmul_fp16_f32out`), not changing the GELU or LN.

Harness (host-side, not a shipped gate): a Python whisper-format weight parser + double golden, a C
driver calling `rocket_encoder_block_fp16` per block, and an env-gated `embd_conv`/`embd_enc` dump
in whisper.cpp's encoder.

## In-model fused integration — measured: byte-identical, but transfer-bound (not a perf win) [HW sweep]
The fused block was wired into whisper.cpp (each encoder layer → one `ggml_map_custom1` →
`rocket_encoder_block_fp16`; build gate `-DWHISPER_ROCKET=ON`, run gate `WHISPER_ROCKET_ENC=1`,
both default off so the stock drop-in is unaffected). The whole encoder — every LayerNorm,
attention, softmax, GELU, residual — then runs on the NPU.

**Correctness: byte-identical transcript at base.en, small.en, AND medium.en** (jfk.wav, vs the
stock-CPU transcript) — the end-to-end confirmation of the cos-0.9998 fidelity above.

**Performance: 3.4–4.4× slower than CPU, and it is transfer-bound, not a perf win.** Encode (ms,
CPU vs fused-NPU): base.en 1677→7332 (4.4×), small.en 6059→~24500 (~4×), medium.en 19606→73828
(3.77×) — the ratio improves with model size but far too slowly to ever reach parity. A per-phase
profile of base.en (`ROCKET_ENC_PROFILE`) attributes the cost: **softmax 46%,
attention total 75%, matmuls only 17%** — i.e. the encoder is dominated by the many small
non-matmul NPU ops repeatedly round-tripping their tensors through host memory, NOT by the matmul
MACs. Proof it is **transfer-bound, not exp-compute-bound**: swapping the on-NPU LUT exp for a
*slower* host `expf` softmax made softmax **3× FASTER** (3636→1217 ms, encode −27%) because it
skips the three NPU jobs (exp/row-sum/scale) and their host↔NPU transfers of the [Tp,Tn] score
matrix. (`ROCKET_ATTN_HOST_SOFTMAX=1` keeps this option.)

**Conclusion: the deep resident-fusion rewrite is not justified for a Whisper perf win.** The
matmul-only offload (the current drop-in) already **beats** the CPU encoder — 1.18× (tiny.en) rising
to 2.14× (large-v3), the win growing with model size ([perf/benchmarks.md](../perf/benchmarks.md) ASR
section, [perf/data/whisper-encoder.md](../perf/data/whisper-encoder.md)) — so this fused whole-block
path (3.4–4.4× *slower*) is strictly worse, and a fully resident fused encoder would at best land near
the drop-in (the per-op readback floor is fundamental — see not-mac-bound.md). The fused path's value is the proven correctness milestone + a substrate
for future fusion. If ever pursued, the lever is to keep ALL intermediates on-NPU across the block
(scores in a BO, on-NPU PPU row-max so softmax needs no host round-trip, cube-resident
`matmul→act→⊙→matmul`, fold scale/bias into the matmul pack; see rmsnorm-onnpu.md, ffn-block.md).

**Softmax is the known NPU transformer bottleneck** (NPUs lack native `exp`): see Sadheerthan et
al., *Attention Distribution-Aware Softmax for NPU-Accelerated On-Device Inference of LLMs*
(Electronics 2026, SOURCES.md). That work optimizes the exp *kernel* (distribution-aware non-uniform
LUT, −18.5% cycles/call) — orthogonal to our dominant cost (transfers), but its quant-domain,
clamp-`[-20,0]`, in-domain-row-max recipe is a good design reference for a fully-on-NPU resident
softmax. Reproduction harness + the whisper patcher: [whisper-encoder-validation/](../whisper-encoder-validation/).
