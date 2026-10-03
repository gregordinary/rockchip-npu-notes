# SigLIP-B/16 encoder on the vendor and mainline stacks

This record holds the same-silicon measurements behind the SigLIP comparison in
[encodings/siglip-encoder.md](../../encodings/siglip-encoder.md) and the SHARD entries in
[SOURCES.md](../../SOURCES.md). The encoder is SmolVLM-256M-Instruct's SigLIP-B/16: twelve
layers, `[1,1024,768]` in and out. Measured 2026-10-03.

## Operating point

Two Turing RK1 modules carry the same RK3588 silicon on two kernels:

| Stack | Kernel and driver | NPU clock | Host |
|---|---|---|---|
| RKNN and SHARD | `6.1.172-vendor-rk35xx`, `rknpu` 0.9.8, `librknnrt` 2.3.2 | 1000 MHz, `rknpu_ondemand` | CPU and DDR governors `performance` |
| rocket | `7.2.8-1-arm64`, `rocket` 1.3.0 | 600 MHz | CPU governor `performance`, `taskset 0xf0` |

Each state is a [live] read on the day. The two NPU clocks differ, so a wall compared across the
two stacks is not iso-clock.

Every RKNN model was built by rknn-toolkit2 2.3.2 with SHARD's own script for that row, from
SHARD's repository at `fe3eefc`. SHARD's board environment pins transformers 4.57.1, and these
builds use the same release.

## Method

The inputs are four real images, one from SHARD's repository and three from COCO val2017. Each
is resized to 512×512, so all 1024 patches are valid and no attention mask applies.

The reference is fp32 PyTorch with identity position ids, which is the model as trained. The
transformers release SHARD pins shifts every position id on a full grid
([encodings/siglip-encoder.md](../../encodings/siglip-encoder.md) §"Configuration"). The
RKNN and SHARD arms take the reference's own embeddings as input, so they see the same
positions either way. The rocket arm computes its own embeddings from the pixels.

Every score is a cosine over all 1024×768 values, in fp64:

- **Encoder output** compares the last residual stream, before the post-LayerNorm.
- **Post-LN** passes both sides through the vision model's `post_layernorm`, the features the
  language model reads.
- **Teacher-forced per block** feeds each of the 24 attention and MLP blocks the reference's
  own input. It is SHARD's validation metric.

The encoder output carries outliers near 700 against a standard deviation near 18. Its cosine
weights them heavily, so it reads high even where the rest is wrong. Post-LN removes that
weighting.

An RKNN or SHARD wall is the mean of 20 `rknnlite` `inference()` calls after five warm-ups,
fp32 in and out. That is SHARD's own timing method (`sbc_run_monolithic.py`). The SHARD rows
time SHARD's `HybridSplitEncoder.forward`: 24 shard calls plus the host scaling. The rocket wall
is the warm median of 20 `rocket_siglip_encode_ctx` calls. That call takes pixels and returns
post-LN features, so it also runs the patch embed and the post-LayerNorm the RKNN rows omit.

## Results

| Arm | Wall | Encoder output cos | Post-LN cos |
|---|---:|---:|---:|
| RKNN fp16, whole encoder, cores 0-2 | 1.195 s | 0.9985-0.9992 | 0.9923-0.9975 |
| RKNN fp16, whole encoder, core 0 | 1.624 s | Same outputs as cores 0-2 | Same as cores 0-2 |
| RKNN int8, SHARD's noise calibration | 1.028 s | 0.521-0.550 | 0.030-0.080 |
| RKNN int8, calibrated on eight COCO images | 1.029 s | 0.587-0.645 | 0.053-0.071 |
| SHARD pack, 24 fp16 shards | 3.600 s | 0.956-0.975 | 0.805-0.878 |
| SHARD pack built without its tiling | 2.125 s | 0.956-0.975 | 0.804-0.878 |
| rocket, simple path (`rocket_siglip_encode`) | Not timed | 0.99997-0.99998 | 0.99981-0.99991 |
| rocket, resident path (`rocket_siglip_encode_ctx`) | 1.394-1.412 s | Not reported | 0.99998-0.99999 |

Each range spans the four images [HW sweep]. The core-0 row was scored on the transformers
4.57.1 embeddings, where its outputs match the three-core build's. Over the 20 calls the RKNN
whole-encoder walls spread by 0.05% or less and the SHARD walls by under 1%. The SHARD pack's
teacher-forced per-block cosine is 0.9991-0.9992 mean, 0.9883-0.9896 at its worst block.

## SHARD's Table 1 rows

| Row | Paper | Same board | What the row measured |
|---|---|---|---|
| RKNN-FP16 | 19.63 s, cos 0.64 | 19.51-19.88 s on `ondemand`, 20.51-20.97 s pinned | The whole `model.generate()` in `run_ablation_tiling.py`, 50 new tokens |
| RKNN-INT8 | 1.40 s, cos 0.02 | 1.028 s, post-LN 0.03-0.08 | The cosine reproduces. The 1.40 s is an fp16 build |
| SHARD | 2.24 s, cos 0.95 | 3.600 s, encoder output 0.956-0.975 | The cosine reproduces as the encoder-output cosine |
| CPU-FP32 | 30 s | Not measured | |

The 19.63 s generate runs SHARD's tiling-ablated pack as the vision encoder and the SmolLM2
language model on the CPU in PyTorch. Run here, it prints SHARD's logged caption word for word.
The same pack, timed alone, takes 2.125 s. One generate runs the encoder once, so the CPU
language model carries most of the wall [hypothesis].

The 1.40 s is SHARD's own `BASELINE_RESULTS` entry for `monolithic_encoder_offload.rknn`, which
`export_partial_offload.py` builds without quantization. That file scores it at cosine 0.99.
The 0.64 appears in no result file or script in the repository [host-computed: a search at
`fe3eefc`].

## Attribution

### The SHARD pack's fidelity loss

SHARD replaces `gelu_pytorch_tanh` with `x·sigmoid(1.702x)`. That substitution alone, run in
fp32 on the host, scores encoder output 0.958-0.976 and post-LN 0.810-0.884
[host-computed]. The NPU pack lands within 0.007 of it on every image, so the NPU adds almost
nothing to the loss.

The teacher-forced metric cannot see it. Every block scores 0.988 or better, and the composed
encoder still loses 0.12-0.20 of post-LN cosine. A per-block cosine on a residual stream bounds
no end-to-end loss.

The vendor compiler lowers the tanh GELU natively (`replace_torch_tanh_gelu` in the build log),
so on rknn-toolkit2 2.3.2 the substitution is unnecessary [source-confirmed: the toolkit's own
build log].

SHARD's sandwich scaling is exact algebra. The host multiplies by 0.1 and the graph by 10 on
the way in, and the reverse on the way out. The block computes the unscaled function, and only
the tensors between shards are scaled.

### The cost of the tiling

SHARD's attention tiling (64 chunks of 16 query rows) is exact math, so the untiled pack scores
identically. Tiling doubles each attention shard, 222-237 ms tiled against 101-112 ms untiled.
The MLP shards stay at 80-84 ms either way [HW sweep]. Over twelve layers that is 1.5 s of wall.

### The vendor fp16 graph

For the whole encoder, rknn-toolkit2 2.3.2 fuses every layer's MatMul, Mul, Softmax and MatMul
into one op (`fuse_matmul_softmax_matmul_to_sdpa`). It also lowers LayerNorm natively
(`replace_torch_layernorm`). That graph never materializes the 1024×1024 score transpose, and
its build emits no CPU-fallback warning. SHARD's `disable_rules` list turns the SDPA fusion off.

The runtime's own profile (`RKNN_QUERY_PERF_DETAIL`, perf collection on, so attribution only)
puts every op but the input and output on the NPU, 320 us of CPU time in a 1.2 s frame:

| Op | Share | Core workload |
|---|---:|---|
| `exSDPAttention`, 12 calls | 59.85% | One core |
| `ConvAdd` (out_proj, fc2), 24 calls | 16.67% | Three cores |
| `ConvExGelu` (fc1 with its GELU), 12 calls | 13.72% | One core |
| `exNorm`, 24 calls | 5.35% | One core |
| `Conv` (q, k, v), 36 calls | 4.02% | Three cores |

The attention runs on one core, as it does in the RKNN whisper-base encoder
([whisper-encoder.md](whisper-encoder.md)).

### The transpose fallback

Only a graph that materializes the score transpose gets the fallback. SHARD's per-layer
baseline adds one on purpose, as `attn_scores + attn_scores.T * 1e-6` (the `score_transpose`
probe). An fp16 build of that layer then prints:

```
Transpose will fallback to CPU, because input shape has exceeded the max limit, height(1024) * width(1024) = 1048576, required product no larger than 8192!
```

The int8 build of the same layer prints `required product no larger than 16384!`. Both limits
are 16 KiB, a bound on bytes rather than elements [host-computed]. The same layer without the probe compiles with no
fallback warning [source-confirmed: rknn-toolkit2 2.3.2 build log, in
[siglip-rknn/transpose-warnings.txt](siglip-rknn/transpose-warnings.txt)].

### The int8 rows

The default int8 build quantizes the graph output too. The toolkit says so:
`The default output dtype of 'out' is changed from 'float32' to 'int8' in rknn model for
performance!` The encoder output spans about ±700, so one per-tensor int8 step is 5.0-5.4.

Quantizing the fp32 reference output alone, with its own range, already leaves post-LN at
0.41-0.46 [host-computed]. The int8 compute takes it the rest of the way, to 0.03-0.08.
Calibrating on real images instead of noise moves post-LN by 0.04 or less, in either direction.

## Reproduce

The scripts, raw outputs and the build-log excerpt are in [siglip-rknn/](siglip-rknn/).
