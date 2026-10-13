# SigLIP-B/16 vision encoder on the NPU

The SigLIP-B/16 vision encoder, the SmolVLM-256M image front end, runs end to end on the NPU
through `rocket`. Its layer is the pre-norm encoder block of the
[Whisper encoder](whisper-encoder.md) (`rocket_encoder_block_fp16`), which runs fully on the
NPU at cos~1. SigLIP is that same block at a new shape, plus a patch embed, a position add and
a post-LayerNorm. It needs no hardware path beyond the block's primitives.

## Lowering

### Configuration (from SmolVLM-256M `vision_config`)

| field | value |
|---|---|
| hidden `d` | 768 |
| layers | 12 |
| heads | 12 (head_dim 64) |
| `d_ff` | 3072 |
| image / patch | 512 / 16 -> L = 1024 patches |
| activation | `gelu_pytorch_tanh` |
| LayerNorm eps | 1e-6 |

A SigLIP encoder layer is structurally identical to the Whisper pre-norm block
(`x += MHA(LN1(x))`, `x += MLP(LN2(x))`, `MLP = fc2(GELU(fc1))`). The only differences are
the shape, and that SigLIP attention has all four q/k/v/o biases (Whisper has no `bk`).
`rocket_encoder_block_fp16(T=1024, d=768, n_head=12, d_ff=3072)` therefore runs a SigLIP layer
directly.

The Idefics3/SigLIP position embedding uses a fractional-coordinate bucketize. For a full
512×512 image (a fully-occupied 32×32 patch grid), it reduces to raster order
`position_ids = arange(1024)`, so the position table is added in patch order.
`siglip_extract.py` bakes the gathered table into the blob, so the C driver just adds
`pos[p]` per patch.

> [!CAUTION]
> transformers 4.55.0-4.57.3 do not reduce to raster order. They compute each coordinate as
> `k / nb * (1 - 1e-6)`, which falls just below its bucket boundary on a full grid. Every
> position id then shifts down one: row and column 0 are used twice, and 31 never. On four real
> images the shifted model's post-LN features score cosine 0.399 against the raster-order model
> [host-computed]. An oracle built on one of those releases fails this encoder by that much.

Releases 4.46-4.53 and 5.0 map a full grid to raster order [source-confirmed:
`modeling_idefics3.py` at each release tag]. SHARD
([SOURCES.md](../SOURCES.md)), which deploys this encoder through rknn-toolkit2, pins 4.57.1 in
its board environment.

### Graph and the host/NPU split

```
pixels[3,512,512] --im2col--> patches[1024,768] --matmul(patch_W^T)+bias+pos--> x[1024,768]
x --12x pre-norm encoder block--> --post-LayerNorm--> out[1024,768]   (SmolVLM consumes these
                                                                       1024x768 patch features;
                                                                       no MAP pooling head)
```

The patch embed is a non-overlapping patchify (stride == kernel == 16, pad 0), so it lowers to
im2col + matmul rather than the conv engine. Each patch's `[ic][kh][kw]` block flattens to a
768-vector, and `x = patches · patch_W^T`. `patch_W` is `patch_embedding.weight` reshaped
`[768, 3·16·16]`, since the flatten order matches the im2col. The shape M=1024, K=768, N=768
satisfies every offload alignment. The im2col gather, the patch bias and the position add
are O(L·d) host glue (the same class as the irreducible host packing, see
[tile-layouts.md](tile-layouts.md)).

### Simple and resident paths

`librocketnpu` implements the encoder as `rocket_siglip_encode`, the simple path, which
re-packs the weights on every call, and `rocket_siglip_encode_ctx`, the resident path
(`rocket-userspace/src/rocket_siglip_encoder.c`). The weight and oracle tooling is
`tools/siglip_extract.py` / `tools/siglip_reference.py`.

Weights are static across images. `rocket_siglip_ctx_create` therefore packs the seven
static GEMMs per layer (patch, q/k/v/o, fc1, fc2) once into resident multicore BOs (~460 ms,
amortized over all images). It uses the prepacked matmul path
([cbuf-reuse.md](cbuf-reuse.md)). Per image, only the activations are packed. LayerNorm,
GELU, and the residual and bias adds run on the host (memory-bound, faster than an NPU
round-trip once the data is already de-tiled). The `1/√d_head` scale is folded into q
(single-stream path) or applied inside the attention kernel (multicore path).

The per-head attention (scores `q_h·k_h^T` -> softmax -> `P·v_h`) fans the 12 heads across
the worker fds by default, and `ROCKET_SIGLIP_FA=0` reverts to the single-stream per-head
loop. The heads are independent (each writes its own output slice), so they split into
contiguous ranges over `nthreads` fds, one drm scheduling entity per fd. The kernel
dispatches the ranges across the NPU cores in parallel. The LLM flash-attention path uses the
same head fan-out ([attention-offload-crossover.md](../perf/attention-offload-crossover.md)).

The fan-out is the encoder's form of "fused attention". It reuses `rocket_flash_attn_fp16_ctx`
(unmasked, n_kv_heads == n_head, plain MHA). Head-chaining batches each worker's per-head QK
matmuls into one job and the AV matmuls into a second. Softmax stays host-side, per worker.

### Weight blob

The weight blob is a 16xint32 header (magic `SGLP`, dims, eps) + the fp16 weights in declaration
order (patch_W, patch_b, pos, per-layer {ln1, q/k/v/o, ln2, fc1, fc2}, post-LN). The C loader
mmaps it and walks the cursor. All linear weights stay row-major `[out,in]` (PyTorch nn.Linear ==
the matmul's `B=[N,K]`), so no transpose is needed.

## Accuracy

**Conditions:** RK3588 (Turing RK1), NPU 600 MHz, 2026-06-24, against an fp32 HF `transformers`
oracle. The input is a 512×512 synthetic image, smooth gradients plus mild noise, seed 1234.

The table gives the per-layer cosine of the on-NPU hidden states against the fp32 HF oracle,
averaged over the 12 layers [HW sweep]. It also gives the post-LayerNorm output:

| path | mean-layer cos | post-LN cos | embeddings cos |
|---|---:|---:|---:|
| simple (`rocket_siglip_encode`, host-handoff FFN) | 0.999983 | 0.999894 | 1.000000 |
| resident (`rocket_siglip_encode_ctx`) | 0.999998 | 0.999987 | 1.000000 |

The accuracy threshold is >=0.99. Embeddings at cos=1.0 confirm that the im2col ordering and
the patch projection are exact. The resident path is more accurate than the simple one
because its LayerNorm runs in host fp32 and its GELU is the exact `gelu_pytorch_tanh` formula.

### Real images

**Conditions:** four real images (one from SHARD, three from COCO), each resized to 512×512,
`7.2.8-1-arm64`, `rocket` 1.3.0.

The real images score the same as the synthetic image to within 0.0001 [HW sweep 2026-10-03].
Both inputs drive the residual stream to about 100 by layer 6, and to 520-700 at the encoder
output:

| path | mean-layer cos | encoder output cos | post-LN cos |
|---|---:|---:|---:|
| simple | 0.999984-0.999988 | 0.99997-0.99998 | 0.99981-0.99991 |
| resident | 0.999997-0.999999 | Not reported | 0.99998-0.99999 |

On the same silicon through the vendor stack, the same four images score post-LN cosine
0.992-0.998 for rknn-toolkit2's fp16 whole-encoder build. SHARD's pack scores 0.80-0.88, and an
int8 build 0.03-0.08 whatever its calibration [HW sweep 2026-10-03]. The method and the
attribution are in [perf/data/siglip-encoder.md](../perf/data/siglip-encoder.md).

### GELU

The GELU form does not limit the accuracy. The simple path uses the block's exact-erf 2-pass
GELU (`x·Φ(x)`). The ~1e-3 erf-vs-tanh difference never drops cosine below five nines, so a
tanh-gate LUT is unnecessary. The resident path computes the exact tanh formula on the host
anyway (free, because the data is already de-tiled there).

## Performance

### Latency

**Conditions:** RK3588 (Turing RK1), mainline 7.1, NPU 600 MHz, `ROCKET_KACC=1`. All CPU cores
are pinned to the `performance` governor, under `taskset 0xf0` (A76 cluster). Each figure is
the warm median of 12 (discard cold).

**Not characterized:** the resident latency with the 2026-10-09 and 2026-10-10
`librocketnpu` changes to this encode.

| path | warm latency |
|---|---:|
| simple (per-call matmul, weights re-packed every call) | ~14 s |
| resident (prepacked weights + multicore + host softmax/GELU) | ~2.71 s |

The resident figure is measured without the head fan-out and the GELU LUT, both on by
default (§"Latency optimizations") [HW sweep]. With both on, the resident warm median is
1.394-1.412 s over four real images, 20 calls each [HW sweep 2026-10-03, `7.2.8-1-arm64`,
`rocket` 1.3.0, 600 MHz, `performance`, `taskset 0xf0`]. That build predates four
`librocketnpu` changes to this encode, from 2026-10-09 and 2026-10-10. They cover the
encoder's scratch and V transpose, its bias add and LayerNorm, the FA's QK de-tile and its
softmax.

On the same silicon through the vendor stack at 1000 MHz, rknn-toolkit2's fp16 whole-encoder
build takes 1.195 s and SHARD's pack 3.600 s [HW sweep 2026-10-03, `6.1.172-vendor-rk35xx`].
Those timings start at the embeddings and stop before the post-LayerNorm, and rocket's include
both. The NPU clocks also differ, so the comparison is not iso-clock. One attention op, on one
core, takes 60% of the RKNN build's frame
([perf/data/siglip-encoder.md](../perf/data/siglip-encoder.md)).

Under the vendor driver, poad42/opennpu_rk3588 runs this encoder with all 48 matmuls on the NPU
and attention, GELU and LayerNorm on the CPU. It reports 995 ms an image on one core. Across
three cores with six or more images in flight it reports 370 ms an image, where one image alone
takes 1181 ms. Its cosine is 0.9999 against the HuggingFace reference [external, not
reproduced, see [../SOURCES.md](../SOURCES.md)].

### Host runtime policy and the CPU governor

**Conditions:** the resident path without the head fan-out and the GELU LUT, NPU core clock
held at 600 MHz throughout.

This workload is host-orchestration-bound (submit `ioctl`, blocking wait on the completion IRQ,
host softmax/LN/GELU, de-tile/readback). The clock patch also re-applies the NPU voltage and
clock over SCMI on every runtime-resume, which is all CPU-side work. CPU frequency and
scheduling therefore set most of the resident warm latency [HW sweep]:

| CPU governor | core placement | resident warm median | run-to-run jitter |
|---|---|---:|---|
| `schedutil` (default) | `taskset 0xf0` | ~5.44 s | ±1.5 s (4.1-7.1 s) |
| `performance` | `taskset 0xf0` | ~2.71 s | ±0.02 s (2.69-2.72 s) |
| `performance` | no taskset | ~3.56 s | ±0.2 s |

Pinning `performance` alone is −50 % and collapses the jitter. A76 placement is a further
~0.85 s. Under `performance`, cold~warm, so a discarded cold run is a governor effect. The
mechanism is in [cpu-governor-and-offload.md](../perf/cpu-governor-and-offload.md), and
[not-mac-bound.md](../perf/not-mac-bound.md) places it in the per-submit overhead.

> [!NOTE]
> Pin the CPU governor before timing this workload, with
> `rocket-userspace/tools/npu_perf_governor.sh performance`.

### NPU clock readback

> [!NOTE]
> The rocket clock patch writes the NPU PVTPLL/SCMI clock directly in `runtime_resume`,
> bypassing the Linux clk framework's cache. `debugfs .../aclk_npu0/clk_rate` (and
> `clk_npu_dsu0`) therefore read a stale 250 MHz even while the cores run at 600 MHz. The
> driver's own `dmesg` line (`core N NPU clk -> 600000000 Hz (reads back 600000000)`) gives the
> rate the cores run at.

### Time breakdown

**Conditions:** the resident path with the heads fanned across the worker fds (the default),
`ROCKET_SIGLIP_PROF=1`, a 2026-06-30 profile. The profile predates the GELU LUT and the
2026-10-09 and 2026-10-10 host-loop changes listed under §"Latency". Absolute ms scale with the
CPU governor and the board load (§"Host runtime policy and the CPU governor").

**Not characterized:** the phase shares on the current library.

With the heads fanned across the worker fds, the per-head QK/softmax/AV collapses into one
multicore block, and the FFN is the next-largest cost. The table gives proportions of the
FA-on encode:

| Phase | Share | Bound |
|---|---|---|
| Attention (scores + softmax + `P·V`) | ~48 % | The multicore flash-attention block (heads across 3 cores, host softmax per worker, head-chaining) |
| GELU | ~16 % as a host `tanhf` | Near-free via a bit-exact fp16->fp16 LUT (below) |
| fc1 + fc2 + q/k/v/o | ~21 % | Prepacked, multicore, bound by de-tile and readback ([not-mac-bound.md](../perf/not-mac-bound.md)), not by packB (that is resident) |
| LayerNorm | ~3 % | |
| Head transpose (re-lay q/k/v to head-major + scatter out) | ~2 % | |
| im2col + pos | ~ negligible | |

The 2026-10-09 and 2026-10-10 changes were measured on Depth Anything v2-small, which runs the
same encoder [HW sweep 2026-10-09 and 2026-10-10, RK1, 600 MHz, `ondemand`]. They cut its
LayerNorm, its bias adds, its head transpose, and its attention's de-tile and softmax.

On a single worker (`ROCKET_SIGLIP_FA=0`) the attention block is ~63 % and ~1.9x slower.
It splits into scores ~15 % + host softmax ~19 % + PV ~27 %. Scores are the most readback-bound
matmul (output `M·N` per head, tiny K=64), so the NEON KACC de-tile (below) helps them most.

Full attention softmax is bound by data movement, not by the call count. Batching the 144
per-head `[1024,1024]` softmaxes into one `[12288,1024]` call per layer does not help on the
NPU (~6.5 s either way). The cost is moving 150M score elements through the EXP-LUT + reduce,
not the submit count. Once the scores are de-tiled to host (they already are, after the
stream matmul), a threaded host softmax is ~10x cheaper and skips a round-trip.

### Latency optimizations

| Optimization | Status |
|---|---|
| NEON KACC de-tile readback | Supported (default) |
| Head fan-out across the worker fds | Supported (default). `ROCKET_SIGLIP_FA=0` reverts to the single-stream per-head loop |
| Bit-exact fp16 GELU LUT | Supported (default). `ROCKET_SIGLIP_GELU_SCALAR=1` reverts to the per-element `tanhf` |
| Cube chaining of the resident FFN | Not supported |
| On-chip GELU | Not supported |
| Resident int8 with a Hadamard rotation | Not characterized on this encoder |

#### NEON KACC de-tile readback

The KACC compute path is `mm_compute_kacc`, the shared primitive every prepacked, stream and
multicore worker runs under `ROCKET_KACC=1`. It reads its output cube back to row-major with
a NEON gather (`detile_store_f16`). The gather is one 128-bit fp16 load + store per 8-column
group, the non-accumulating fp16->fp16 sibling of the single-fd `detile_accum_f16`. It is
bit-identical to a scalar per-element gather, and generalizes the de-tile to the resident path.

Against the scalar gather, it cuts SigLIP resident warm ~3.06 -> ~2.71 s (−10 %),
concentrated in the readback-bound attention scores (~0.81 -> ~0.51 s). The same primitive
backs the Whisper encoder and the LLM prefill prepacked path, so the gain applies to any
readback-bound KACC matmul. The gain is largest where output `M·N` is big and K is small, and
~flat on compute-bound prefill.

#### Head fan-out

The head fan-out (§"Simple and resident paths") cuts the attention block ~1.9x and the whole
resident encode 1.44-1.51x warm (cosine identical, 0.999998 mean-layer) [HW sweep, 600 MHz,
loaded board]. On a single worker, the attention is the resident path's largest slice. The
fan-out reuses the LLM flash-attention primitive rather than a new kernel, so the
head-chaining and resident-scratch work applies to it directly.

The score `L×L` cube still round-trips host-side for the host softmax. The online/tiled
FA-2 variant that avoids that is a dispatch-bound loss here, as on the LLM path. At a ViT
encoder's shape, the host passes over the scores take about three quarters of a worker's head
range. The device waits the rest. The split is in
[attention-offload-crossover.md](../perf/attention-offload-crossover.md) §"The head range's
host passes at an encoder shape".

#### Bit-exact fp16 GELU LUT

fp16 GELU is a function of a 16-bit value, so all 65536 outputs fit one 128 KB table
(`g_gelu_lut`, built once over every fp16 bit pattern incl. inf/nan). The host GELU then
turns its per-element `tanhf` into a load. The load is bit-identical to the scalar path (it
tabulates the exact same `gelu_pytorch_tanh`), so cosine is unchanged. With attention
multicored, GELU is the second-largest cost (§"Time breakdown"). The LUT is 1.22x warm on the
whole resident encode.

The fan-out and the LUT together are 1.78x warm (4.10 -> 2.30 s, loaded board, cosine
0.999998) [HW sweep, 600 MHz]. The resident encode then runs under the 2.71 s idle baseline
even under load.

An on-chip GELU would need the cube residency that loses multicore (§"Cube chaining of the
resident FFN"). An isolated NPU activation of the `[1024,3072]` intermediate is a
pack/readback round-trip. The host LUT is a load, with no round-trip, and bit-exact.

#### Cube chaining of the resident FFN

A cube-chained resident FFN is slower than the multicore host-handoff FFN [HW sweep
2026-06-29]. On the resident path, fc1 and fc2 are prepacked multicore (3 cores). A
cube-resident `fc1 -> +bf1 -> GELU -> fc2` chain is inherently single-fd (the consumer matmul
reads the producer's one cube BO, one IOVA), so it forfeits that 3-core parallelism. The lost
multicore outweighs the saved host GELU and intermediate de-tile/re-tile.

Cube-chaining and multicore are therefore mutually exclusive, and on the resident path
multicore is faster. The gain from cube chaining is on the simple (single-fd) path, where
`rocket_mlp_fp16_fused` in `rocket_encoder_block_fp16` reduces the pack and read time. The
measurements are in [cross-op-chaining.md](cross-op-chaining.md) and
[ffn-block.md](ffn-block.md). An `LN -> matmul` chain is single-fd in the same way
[expected].

#### Resident int8 with a Hadamard rotation

Resident int8 and the Hadamard rotation exist in the library, and reduce memory use. On this
readback-bound workload the int32 readback is expected to hold it near fp16 speed (the LLM
result, [k-accumulation.md](k-accumulation.md)) [expected]. It is a fidelity and memory option,
not a latency one.

## Evidence

### Accuracy

The accuracy test is `siglip_rocket` (`rocket-userspace/tests/siglip_rocket.c`). Its synthetic
input image comes from `siglip_reference.py` (seed 1234).

### Reproduction

The commands below run the accuracy test and the resident latency bench:

```
# on the board (venv with torch+transformers; model + artifacts under $MODELS, a large disk):
python tools/siglip_reference.py --out $MODELS/siglip/artifacts                # fp32 oracle
python tools/siglip_extract.py   --out $MODELS/siglip/artifacts/siglip_weights.f16
ctest --test-dir build_nv -R siglip_rocket                                   # fidelity gate
sudo rocket-userspace/tools/npu_perf_governor.sh performance                 # REQUIRED for representative latency
ROCKET_KACC=1 ROCKET_SIGLIP_BENCH=12 taskset 0xf0 ./build_nv/siglip_rocket   # +resident bench
#   ROCKET_SIGLIP_PROF=1 adds the per-phase breakdown
sudo rocket-userspace/tools/npu_perf_governor.sh schedutil                   # restore when done
```
