# Whisper encoder on the NPU

The pieces below are validated on hardware on the RK1 at 600 MHz against an fp64 oracle
(gates `exp_lut_rocket`, `softmax_rocket`, `layernorm_rocket`, `conv1d_rocket`, `mha_rocket`,
`encoder_block_rocket`). They compose the full Whisper/transformer encoder block. The
section on the whole-graph vendor compile carries its own operating point.

## EXP LUT (`ROCKET_ACTIVATION_EXP`, enum 10)

Softmax's `exp` is the DPU LUT's EXP kind over `[-16,0]`, whose geometry, accuracy and q=0
table-entry fault are in [dpu-lut-activation.md](dpu-lut-activation.md) §"EXP, the softmax
numerator".

## Softmax (`rocket_softmax_fp16`, src/rocket_softmax.c)

Softmax runs row-wise over the last axis. The composition is:

1. **Host**: row-max + subtract (-> EXP's <=0 domain).
2. **NPU**: `exp` (LUT).
3. **NPU**: row-sum (feature-axis ones-matmul reduce, fp32).
4. **Host**: `1/s` (O(M), exact).
5. **NPU**: per-row scale (`scale_rows`=ew_mul).

The row-max is mandatory. The matmul and the feature-reduce can only sum, never max, and the
only on-NPU max is the PPU max-pool, the resident-fusion path. Validated to T=1500 (Whisper
seq): rows sum to 1±0.0005, max_abs ~1e-4.

## LogSoftmax (`rocket_logsoftmax_fp16`, src/rocket_softmax.c)

`out = x − logsumexp(x)` is the classification and NLL-loss head, and the LM log-prob
output. It shares softmax's steps 1-3 (host row-max+subtract -> NPU `exp` -> NPU row-sum
`s`). The tail is then a subtract of a per-row scalar instead of a divide. The host computes
`ls = log(s)` (O(M), exact), and the NPU runs a per-row broadcast `ew_sub`
(`out = (x−rowmax) − ls`). The row-max term contributes `exp(0)=1`, so `s>=1`, `ls>=0` and
`out<=0`.

The per-row log stays on the host, not on the LOG LUT. It is M scalars already host-side as
fp32 after the reduce read-back, so host `log` is exact and adds no round-trip. Softmax
keeps `1/s` on the host the same way, and RMSNorm keeps `rsqrt`. The DPU
[`ROCKET_ACTIVATION_LOG`](dpu-lut-activation.md) LUT is for a large tensor of logs, a
different use.

LogSoftmax is all-additive, so it is better-conditioned than softmax for a gate (no
tiny-probability relative blow-up). Check absolute error. On hardware: `max_abs <=0.031`
(the wide-spread amp=20 case, ~0.008 typical = fp16 storage), `Σ_n exp(out)=1`, validated
to T=1500.

## Stable cross-entropy (`rocket_cross_entropy_fp16`, src/rocket_softmax.c)

`CE[m] = logsumexp(logits[m]) − logits[m][target[m]] = −logsoftmax(logits[m])[target[m]]` is
the softmax-classifier and LM NLL loss, one scalar per row. This is its numerically-stable
form: it never materializes softmax and has no divide. It reuses the LogSoftmax front half
(the logsumexp reduction):

1. **Host**: row-max + subtract.
2. **NPU**: `exp` (LUT).
3. **NPU**: fp32 row-sum `s`.
4. **Host**: `lse = rowmax + log(s)`.
5. **Host gather**: `logits[m][target[m]]`.
6. `CE[m] = lse − gathered`.

Cross-entropy is therefore the on-NPU logsumexp plus a host gather and subtract. It is
strictly cheaper than LogSoftmax: it skips the tail per-row `ew_sub`, because only the M
target log-probs are needed, not the full `[M,N]` output.

There is no hardware gather on the RK3588 NPU. No indexed/scatter read path exists
(the same fact the [reduce](feature-reduce.md) / attention notes record), so `logits[m][target[m]]`
is M scalar host index-lookups, correct and free, exactly like softmax's host `1/s` and LogSoftmax's
host `log(s)`. A gather is not a contraction or a pool, so neither the matmul nor the PPU supplies it.

CE is fp32-grade and more accurate than LogSoftmax (hardware `max_abs <= 1.5e-4` vs 0.031).
The loss is `lse − gathered`, both fp32/host values. The only NPU-introduced error is in `s`
(the fp32 sum of the LUT-`exp`). The host takes `log(s)` in fp64, so the error is just
`log(s_npu/s_ref) ~ 1e-4`. The loss never round-trips through fp16 output storage. That
fp16 storage, of the very-negative log-probs, is the term that dominates LogSoftmax's
error, and CE skips it.

The loss invariant is `s >= 1`, so `log(s) >= 0`, so `CE >= 0`. The gate checks it, with one
`abs_tol` of LUT slack at the argmax row where `CE = log(s) ~0`. The gate
`tests/cross_entropy_rocket.c` compares against an fp64 CE oracle across:

- The M-tile boundary
- N%32≠0
- A T=1500 vocab
- Random + forced-argmax targets
- A wide spread

The host self-check `CE == −logsoftmax_ref[target]` cross-checks against the independent
LogSoftmax path. An optional follow-on is a `softmax_cross_entropy` that also returns the
softmax (backward grad `softmax − onehot`). It must keep the loss path stable and not
divide.

## LayerNorm (`rocket_layernorm_fp16`, src/rocket_norm.c)

LayerNorm computes `(x-mean)/sqrt(var+eps)*gamma + beta` per row. Both reductions share one
feature-reduce job by stacking rows: A = `[x ; x⊙x]` (2M rows) under the ones weight. The
first M outputs are `sum(x)` and the next M are `sum(x²)`. A second ones-column does not
give `sum(x²)`: N-columns are weighted sums of the same A, i.e. of x, not x², so stack rows.

The tail is O(M) on the host (mean,var,rsqrt). The affine folds to `out = x⊙A + B`,
A=r·gamma, B=beta−mean·r·gamma (one ew_mul + one ew_add). The fp16-square overflow prescale
is RMSNorm's ([rmsnorm-onnpu.md](rmsnorm-onnpu.md)). The beta argument can be NULL.

## conv1d front-end (`rocket_conv1d_fp16`, src/rocket_conv.c)

Whisper's two front-end convs (kw=3, pad=1, conv1 IC=n_mels stride 1, conv2 stride 2) are a
width-only 1D conv, lowered onto the validated `rocket_conv2d_fp16`. **Lower with time on
the height axis (IW=1), not width.** The natural width-on-time layout (IH=1) overflows the
feature banks. The conv tiler tiles output rows (oh) first, so a long or many-channel
sequence shrinks the per-tile height to fit one CBUF pass. The width-on-time layout leaves
oh=1 untileable and overflows the feature banks for Whisper IC=80/512 (gen returns -1). The
layouts are byte-identical either way (no repack).

Small reductions are bit-exact. Large IC·kw is within fp16-accum tolerance: the CNA accum
order differs from the host loop, at 1-ULP store diffs, which is benign.

## Multi-head self-attention (`rocket_mha_self_fp16`, src/rocket_attn.c)

This is the pure attention sublayer (no LN/residual). Every stage below runs on the NPU:

| Stage | Computation |
|---|---|
| q/k/v | x·W^T (+bias), an NPU matmul |
| Per-head scores | scale·(q_h·k_h^T) |
| Softmax | row-wise |
| Per-head context | ctx_h = P·v_h |
| Output | out = concat·Wo^T+bo |

The host glue is head slicing, the d_head^-0.5 scale, the bias adds and the per-head V
transpose. The matmul needs B as [N,K], so ctx = matmul(P, v_h^T) = P·v_h.

### Key-count alignment

`rocket_matmul_fp16` rejects unaligned N/K (for example N=100). The key count T is the N
of the scores matmul and the K of ctx. So **pad the key count to Tn=(T+31)&~31 and mask the pad
score columns to −30000 before softmax**. Zero pad keys give score 0, which softmax would
weight as exp(0). Masking drives exp to underflow (~0), and the zero pad V rows add nothing
to ctx.

Query rows pad to Tp=(T+3)&~3 (M%4). This makes attention correct for any T (Whisper T=1500
is itself unaligned). The result is cos=1.000000 vs the fp64 oracle at Whisper-base
d=512/8-head, including T%16≠0.

## Encoder block (`rocket_encoder_block_fp16`, src/rocket_encoder.c)

The block is Whisper pre-norm: `x = x + MHA(LN1(x)); x = x + MLP(LN2(x))`, with MLP =
`GELU(h·Wf1^T+bf1)·Wf2^T+bf2`. It runs on the NPU apart from the host steps named above
(softmax's row-max and `1/s`, LayerNorm's per-row tail, the attention glue and bias adds):

- Both LayerNorms
- All attention matmuls and every softmax
- Both residual adds
- The two MLP projection matmuls
- The MLP's GELU

The result is cos=1.000000 vs the fp64 block oracle (d=256/512, including T%16≠0). The
max_abs is ~2e-3, which is fp16 quant at output magnitude ~2.4.

## On-NPU GELU: the 2-pass `x·Φ(x)`

GELU runs the 2-pass route (like SiLU): `GELU(x)=x·Φ(x)`, where Φ is the Gaussian CDF. The
CDF is a monotone `[0,1]` function on the clean unit-LUT geometry (no QUIRK-1 flat-region
spike), followed by a DPU EW-mul by x. In the encoder MLP, `Φ(f1)` runs on the DPU LUT
(`ROCKET_ACTIVATION_GELU_GATE`), then `f1·Φ(f1)` on the DPU EW-mul. Validated on hardware:
cos=1.000000 vs true erf-GELU over `[-12,12]` (`tests/gelu_rocket.c`).

A fused single-pass matmul->GELU (1×1-conv-act epilogue) fails for wide FFN inputs, at
cos~0.04 and max_abs=128. The QUIRK-1 flat-tail mux spike hits en masse (fc1 output spans
`GELU(x)~0`, x≲-3). **Single-pass LUT fusion is curved-region-only.** The durable on-NPU
GELU/SiLU is the 2-pass `x·gate(x)`.

## Validation against whisper.cpp (real base.en, real audio)

The composition above is validated against whisper.cpp's real encoder on real audio
(jfk.wav), as well as against the self-contained fp64 block oracle [HW sweep]. The model is
base.en (d=512, 8 heads, d_ff=2048, 6 blocks, T=1500). The method:

1. Dump whisper's encoder input (`embd_conv` + positional embedding) and final output
   (`embd_enc`, post-LN) from a stock-CPU run.
2. Feed the same input and the real model weights to `rocket_encoder_block_fp16` on the NPU.
3. Compare per block and end to end.

A double-precision reference of whisper's exact math sits between the two as the golden
(biased-variance LayerNorm eps=1e-5, tanh-approx GELU, attention scale 1/sqrt(d_head), K
has no bias). The golden is itself confirmed == whisper (`embd_enc` cos 0.99983). The
results:

- **Per-block, fed the golden input: cos 1.000000** (isolated, so the block computation is
  faithful).
- **Chained through all 6 blocks: cos >= 0.99997.** The final post-LN encoder output is
  cos 0.99981 vs whisper's `embd_enc`, far past the >=0.99 bar (SHARD, a SOTA RK3588 VLM,
  runs at 0.95).
- **The erf-vs-tanh GELU gap is a non-issue.** Raw |tanh-GELU − erf-GELU| peaks at 4.7e-4
  (x~2.7). At the block level the two are cos 1.0 / max_abs ~1.7e-4 (within fp16 noise).
  The block's erf GELU needs no tanh-approx variant to match whisper.

### Whisper activation outliers and the LayerNorm fp16-square prescale

On real audio the encoder develops a few stable, dominant outlier channels [HW sweep]. In
base.en, channel 270 reaches |x|~1221 at every block, and channels 67/33/64/73/509 reach
~90-200. The residual-stream RMS jumps from ~0.9 (block 3) to ~8.4 (block 4) as most
channels grow large. The consequences:

- **fp16 x² overflows.** Channel 270 squared is ~1.5e6 > fp16 max (65504), so a naive fp16
  LayerNorm variance sum would be Inf. `rocket_layernorm_fp16`'s power-of-2 prescale
  ([rmsnorm-onnpu.md](rmsnorm-onnpu.md)) keeps the encoder finite and correct here. It
  fires on every real Whisper block, not as a corner case.
- **Outliers amplify fp16 chaining error.** The next block's high-gain weights amplify a
  tiny fp16 difference on an outlier channel. The intermediate rocket-vs-whisper max_abs
  therefore grows to ~48 by block 5 (RMS ~8.6) while cosine stays >=0.99997. Post-LN
  re-normalizes the magnitude away, so the encoder output is faithful (cos 0.9998). The
  single large surviving element post-LN (max_abs ~6.7) is the outlier channel scaled by
  its LN gamma.
- **The residual stream is the precision floor, not the matmul.** Tightening past
  cos 0.9998 (if ever needed) means carrying the residual/FFN in higher precision (the
  fp32-output matmul `rocket_matmul_fp16_f32out`), not changing the GELU or LN.

The harness is host-side, not a shipped gate. It is a Python whisper-format weight parser +
double golden, a C driver calling `rocket_encoder_block_fp16` per block, and an env-gated
`embd_conv`/`embd_enc` dump in whisper.cpp's encoder.

## In-model fused integration

The fused block is wired into whisper.cpp and measured on hardware [HW sweep]. Each encoder
layer becomes one `ggml_map_custom1`, which calls `rocket_encoder_block_fp16`. The build
gate is `-DWHISPER_ROCKET=ON` and the run gate is `WHISPER_ROCKET_ENC=1`. Both default off,
so the stock drop-in is unaffected. Every LayerNorm, attention, softmax, GELU and residual
then runs on the NPU, with the host steps that §"Encoder block" names.

### Correctness

The transcript is byte-identical at base.en, small.en, and medium.en (jfk.wav, vs the
stock-CPU transcript). That is the end-to-end confirmation of the cos-0.9998 fidelity above.

### Performance

The fused path is 3.4-4.4x slower than the CPU. It is transfer-bound, and it is not a perf
win. Encode time:

| Model | CPU (ms) | Fused NPU (ms) | Ratio |
|---|---:|---:|---:|
| base.en | 1677 | 7332 | 4.4x |
| small.en | 6059 | ~24500 | ~4x |
| medium.en | 19606 | 73828 | 3.77x |

The ratio improves with model size, but far too slowly to ever reach parity. A per-phase
profile of base.en (`ROCKET_ENC_PROFILE`) attributes the cost: softmax 46%, attention total
75%, matmuls only 17%. The many small non-matmul NPU ops dominate the encoder, because they
repeatedly round-trip their tensors through host memory. The matmul MACs do not dominate it.

The path is transfer-bound, not exp-compute-bound. Swapping the on-NPU LUT exp for a slower
host `expf` softmax made softmax 3x faster (3636->1217 ms, encode −27%). The host softmax
skips the three NPU jobs (exp/row-sum/scale) and their transfers of the [Tp,Tn] score
matrix between host and NPU. `ROCKET_ATTN_HOST_SOFTMAX=1` keeps this option.

This fused path is not a Whisper perf win. The matmul-only offload (the current drop-in)
beats the CPU encoder at 1.18x (tiny.en), rising to 2.14x (large-v3), with the win growing
with model size. See the ASR section of [perf/benchmarks.md](../perf/benchmarks.md) and
[perf/data/whisper-encoder.md](../perf/data/whisper-encoder.md). The fused whole-block path
(3.4-4.4x slower) is therefore strictly worse.

The per-op readback floor ([not-mac-bound.md](../perf/not-mac-bound.md)) bounds any
composition that returns each intermediate to the host. It does not bound a resident
encoder, and a whole-graph compile runs 1.70x faster than the drop-in on the same board.

### The whole-graph vendor compile

On one board, a whole-graph vendor compile runs the 20 s whisper-base encoder 1.70x faster
than the drop-in. The RKNN encoder takes 344 ms at 1000 MHz on three cores, and the drop-in
585 ms, against 830 ms on the CPU. At 600 MHz the ratio is 1.60x, 364 ms against 581 [HW sweep,
Turing RK1, `6.1.172-vendor-rk35xx`, `rknpu` 0.9.8, 2026-09-26].

The RKNN arm is `rknn_model_zoo`'s fp16 encoder, compiled by rknn-toolkit2 2.3.0 and run
through `librknnrt`. The drop-in is stock `whisper-cli -ac 1000` with ggml-rocket through rknpu-submit. The
encoder at 1000 positions is 52.1 GFLOP, so the RKNN graph runs at ~151 GFLOP/s here and the
drop-in at ~89. The comparison is timing only: the two encoders' outputs were not compared.

The RKNN encoder's wall is its fused attention op. `exSDPAttention` runs on one core whatever
the core mask, and it takes 240 ms of the 345 ms per-op sum that `RKNN_QUERY_PERF_DETAIL`
reports on three cores. Only the matmul-type ops
split across cores, so three cores buy 14% over one, 344 against 400 ms. No op runs on the CPU.

The attention op slows 1.56x from 1000 MHz to 300, but only 1.06x from 1000 to 600, and the frame
follows it. DDR frequency does not bind it: DDR stays at 2112 MHz through the run, and pinning
it there changes nothing. The bound above 600 MHz is not located. A clock the NPU devfreq
does not scale, such as the NPU's bus clock, would produce this shape [hypothesis].

Seeed's ~250 ms for the same encoder is not reproduced on this board ([SOURCES.md](../SOURCES.md)).
Their model file with their runtime version, 2.3.0, on all three cores at 1000 MHz reads 344 ms,
and 2.3.2 reads the same. The raw figures are in
[perf/data/whisper-encoder.md](../perf/data/whisper-encoder.md).

### The lever for a faster fused path

The fused path's value is the proven correctness milestone and a substrate for future
fusion. If the fused path is pursued, the lever is to keep all intermediates on-NPU across
the block:

- Scores in a BO
- An on-NPU PPU row-max, so softmax needs no host round-trip
- A cube-resident `matmul->act->⊙->matmul`
- Scale and bias folded into the matmul pack

See rmsnorm-onnpu.md and ffn-block.md.

Softmax is the known NPU transformer bottleneck (NPUs lack native `exp`): see Sadheerthan et
al., *Attention Distribution-Aware Softmax for NPU-Accelerated On-Device Inference of LLMs*
(Electronics 2026, SOURCES.md). That work optimizes the exp kernel (distribution-aware
non-uniform LUT, −18.5% cycles/call), which is orthogonal to the dominant cost here
(transfers). Its quant-domain, clamp-`[-20,0]`, in-domain-row-max recipe is a good design
reference for a fully-on-NPU resident softmax. The reproduction harness and the whisper
patcher are in [whisper-encoder-validation/](../whisper-encoder-validation/).
