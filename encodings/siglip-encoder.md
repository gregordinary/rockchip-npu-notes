# SigLIP-B/16 vision encoder on the NPU — SmolVLM-256M front-end, head-to-head vs SHARD

The **SigLIP-B/16 vision encoder** (the SmolVLM-256M image front-end) runs end-to-end on the
FOSS rocket NPU. Driver `rocket_siglip_encode` / `rocket_siglip_encode_ctx`
(`rocket-userspace/src/rocket_siglip_encoder.c`),
gate `siglip_rocket` (`rocket-userspace/tests/siglip_rocket.c`),
weight + oracle tooling `tools/siglip_extract.py` / `tools/siglip_reference.py`. HW-validated on
the RK1 at 600 MHz against an fp32 HF `transformers` oracle (measured 2026-06-24).

This is plumbing over primitives that already existed, not new hardware RE: the pre-norm encoder
block was already fully on-NPU at cos≈1 ([whisper-encoder.md](whisper-encoder.md),
`rocket_encoder_block_fp16`). SigLIP is that same block at a new shape plus a patch embed, a
position add, and a post-LayerNorm.

## Configuration (from SmolVLM-256M `vision_config`)

| field | value |
|---|---|
| hidden `d` | 768 |
| layers | 12 |
| heads | 12 (head_dim 64) |
| `d_ff` | 3072 |
| image / patch | 512 / 16 → **L = 1024 patches** |
| activation | `gelu_pytorch_tanh` |
| LayerNorm eps | 1e-6 |

A SigLIP encoder **layer is structurally identical to the Whisper pre-norm block**
(`x += MHA(LN1(x))`, `x += MLP(LN2(x))`, `MLP = fc2(GELU(fc1))`) — the only differences are the
shape and that SigLIP attention carries **all four q/k/v/o biases** (Whisper has no `bk`). So
`rocket_encoder_block_fp16(T=1024, d=768, n_head=12, d_ff=3072)` runs a SigLIP layer directly.

The Idefics3/SigLIP position embedding uses a fractional-coordinate bucketize that, for a full
512×512 image (a fully-occupied 32×32 patch grid), reduces to raster order `position_ids =
arange(1024)` — i.e. the position table is added in patch order. `siglip_extract.py` bakes the
gathered table into the blob so the C driver just adds `pos[p]` per patch.

## Graph and the host/NPU split

```
pixels[3,512,512] --im2col--> patches[1024,768] --matmul(patch_W^T)+bias+pos--> x[1024,768]
x --12x pre-norm encoder block--> --post-LayerNorm--> out[1024,768]   (SmolVLM consumes these
                                                                       1024x768 patch features;
                                                                       no MAP pooling head)
```

The patch embed is a non-overlapping patchify (stride == kernel == 16, pad 0), so it lowers to
**im2col + matmul** rather than the conv engine: each patch's `[ic][kh][kw]` block flattens to a
768-vector and `x = patches · patch_W^T` (`patch_W` is `patch_embedding.weight` reshaped
`[768, 3·16·16]` — the flatten order matches the im2col). M=1024, K=768, N=768 satisfies every
offload alignment. The im2col gather + patch-bias + position add are O(L·d) host glue (the same
class as the irreducible host packing — [tile-layouts.md](tile-layouts.md)).

## Fidelity [HW sweep] — the clean, hardware-independent axis

Per-layer cosine of the on-NPU hidden states vs the fp32 HF oracle, averaged over the 12 layers
(SHARD's metric), plus the post-LayerNorm output, on one 512×512 image:

| path | mean-layer cos | post-LN cos | embeddings cos |
|---|---:|---:|---:|
| simple (`rocket_siglip_encode`) | **0.999983** | 0.999894 | 1.000000 |
| resident (`rocket_siglip_encode_ctx`) | **0.999998** | 0.999987 | 1.000000 |

vs **SHARD 0.95**, RKNN-FP16 0.64, RKNN-INT8 0.02. The fidelity target (≥0.99) is cleared by four
nines. Embeddings at cos=1.0 confirms the im2col ordering + patch projection are exact. The
resident path is *more* accurate than the simple one because its LayerNorm runs in host fp32 and
its GELU is the exact `gelu_pytorch_tanh` formula.

**GELU is a non-issue.** The simple path uses the block's exact-erf 2-pass GELU (`x·Φ(x)`); the
≈1e-3 erf-vs-tanh difference never drops cosine below five nines, so the planned tanh-gate LUT is
unnecessary. The resident path computes the exact tanh formula on the host anyway (free, it is
already de-tiled there).

## Latency [HW sweep] — indicative, NOT iso-hardware vs SHARD

RK1, mainline 7.1, NPU @ 600 MHz, `ROCKET_KACC=1`, **all CPU cores pinned to the `performance`
governor**, `taskset 0xf0` (A76 cluster), warm median of 12 (discard cold):

| path | warm latency |
|---|---:|
| simple (per-call matmul, weights re-packed every call) | ~14 s |
| resident (prepacked weights + multicore + host softmax/GELU) | **~2.71 s** |

vs **SHARD 2.24 s** on an Orange Pi 5 Max (RKNN, unknown NPU clock). **Different board / kernel /
driver / clock → latency is indicative only; cosine is the apples-to-apples comparison.**

### Host runtime policy is the dominant latency lever, not NPU clock

This workload is host-orchestration-bound (submit `ioctl`, blocking wait on the completion IRQ,
host softmax/LN/GELU, de-tile/readback), and the clock patch re-applies the NPU V+clock over SCMI
on **every** runtime-resume — all CPU-side work. So the resident warm latency is set mostly by CPU
frequency/scheduling, with the NPU core clock held fixed at 600 MHz throughout:

| CPU governor | core placement | resident warm median | run-to-run jitter |
|---|---|---:|---|
| `schedutil` (default) | `taskset 0xf0` | ~5.44 s | ±1.5 s (4.1–7.1 s) |
| `performance` | `taskset 0xf0` | **~2.71 s** | ±0.02 s (2.69–2.72 s) |
| `performance` | no taskset | ~3.56 s | ±0.2 s |

Pinning `performance` alone is **−50 %** and collapses the jitter; A76 placement is a further
~0.85 s. The "discard cold run" rule is itself a governor artifact — under `performance` cold≈warm.
Use `rocket-userspace/tools/npu_perf_governor.sh performance`
before benching; this is the on-our-path measurement of the CPU-governor floor that
[not-mac-bound.md](../perf/not-mac-bound.md) flags. Beating SHARD's 2.24 s from here is a structural
problem (the levers below), not a tuning one.

**Clock-readback trap:** the rocket clock patch writes the NPU PVTPLL/SCMI clock directly in
`runtime_resume`, bypassing the Linux clk framework's cache — so `debugfs .../aclk_npu0/clk_rate`
(and `clk_npu_dsu0`) read a **stale 250 MHz even while the cores run at 600 MHz**. Ground truth is
the driver's own `dmesg` line (`core N NPU clk -> 600000000 Hz (reads back 600000000)`), not the
clk debugfs node.

### The resident path

Weights are static across images, so the seven static GEMMs per layer (patch, q/k/v/o, fc1, fc2)
are packed **once** into resident multicore BOs at `rocket_siglip_ctx_create` (≈460 ms, amortized
over all images) via the prepacked matmul path ([cbuf-reuse.md](cbuf-reuse.md)). Per
image only the activations are packed. LayerNorm, GELU, and the residual/bias adds run on the
host (memory-bound, faster than an NPU round-trip once the data is already de-tiled). The
`1/√d_head` scale is folded into q (single-stream path) or applied inside the attention kernel
(multicore path).

The per-head attention (scores `q_h·k_h^T` → softmax → `P·v_h`) **fans the 12 heads across the
worker fds** (default; `ROCKET_SIGLIP_FA=0` reverts to the single-stream per-head loop). The heads
are independent — each writes its own output slice — so they split into contiguous ranges over
`nthreads` fds, one drm scheduling entity per fd, and the kernel dispatches the ranges across the
NPU cores in parallel (the same head-fan-out as the LLM flash-attention path,
[attention-offload-crossover.md](../perf/attention-offload-crossover.md)). This is the realization
of "fused attention" for the encoder: it reuses `rocket_flash_attn_fp16_ctx` (unmasked, n_kv_heads
== n_head — plain MHA; head-chaining batches each worker's per-head QK matmuls into one job and the
AV matmuls into a second; softmax stays host-side, per worker). It cuts the attention block ~1.9× and
the whole resident encode **1.44–1.51× warm** (cosine identical, 0.999998 mean-layer) [HW sweep,
600 MHz, contended box] — the attention was the resident path's largest slice and had been running
on a single worker.

### Where the time goes (resident, multicore attention default, `ROCKET_SIGLIP_PROF=1`)

With the heads fanned across the worker fds (the default), the per-head QK/softmax/AV collapses into
one multicore block and the **FFN is now the next-largest cost**. Proportions of the FA-on encode
(absolute ms scale with the CPU governor + box load; see the governor note above):

- **attention (scores + softmax + `P·V`) ≈ 48 %** — the multicore flash-attention block (heads across
  3 cores, host softmax per worker, head-chaining). On a single worker (`ROCKET_SIGLIP_FA=0`) this same
  block was ~63 % and ~1.9× slower: scores ~15 % + host softmax ~19 % + PV ~27 %. Scores are the most
  readback-bound matmul (output `M·N` per head, tiny K=64), so the NEON KACC de-tile (below) helps them
  most.
- **GELU** — was the #2 cost (~16 %) as a host `tanhf`; now **near-free** via a bit-exact fp16→fp16 LUT
  (below). Cube-resident GELU (fusing fc1→GELU→fc2) is single-fd, which forfeits the multicore fc1/fc2 —
  a net loss here (lever #1 below).
- **fc1 + fc2 + q/k/v/o ≈ 21 %** — prepacked, multicore; **de-tile/readback bound**, the not-mac-bound
  floor ([not-mac-bound.md](../perf/not-mac-bound.md)), not packB (that is resident).
- LayerNorm ≈ 3 %, head transpose (re-lay q/k/v to head-major + scatter OUT) ≈ 2 %, im2col + pos ≈
  negligible.

**NPU FACT — full attention softmax is data-movement bound, not per-call bound.** Batching the 144
per-head `[1024,1024]` softmaxes into one `[12288,1024]` call per layer did *not* help on the NPU
(≈6.5 s either way): the cost is moving 150M score elements through the EXP-LUT + reduce, not the
submit count. Once the scores are de-tiled to host (they already are, after the stream matmul), a
threaded host softmax is ~10× cheaper and skips a round-trip.

## Latency levers

**Shipped — NEON KACC de-tile readback.** The KACC compute path (`mm_compute_kacc`, the shared
primitive every prepacked/stream/multicore worker runs under `ROCKET_KACC=1`) read its output cube
back to row-major with a scalar per-element gather. It now uses a NEON gather (`detile_store_f16`:
one 128-bit fp16 load + store per 8-column group, the non-accumulating fp16→fp16 sibling of the
single-fd `detile_accum_f16`), bit-identical to the scalar gather it replaces. This is the
de-tile generalized to the resident path. It cut SigLIP resident warm ~3.06 → ~2.71 s (−10 %),
concentrated in the readback-bound attention scores (~0.81 → ~0.51 s). The same primitive backs
the Whisper encoder and the LLM prefill prepacked path, so the win carries to any readback-bound
KACC matmul (largest where output `M·N` is big and K is small; ~flat on compute-bound prefill).

**Shipped — multicore attention (head fan-out).** The resident-path "fused attention" lever is the
head fan-out described under *The resident path* above: `rocket_flash_attn_fp16_ctx` runs the 12 heads
across the worker fds (default; `ROCKET_SIGLIP_FA=0` reverts to the single-stream per-head loop), 1.44–
1.51× warm on the whole encode, bit-faithful. It reuses the LLM flash-attention primitive rather than a
new kernel, so the head-chaining + resident-scratch work carries straight over. The score `L×L` cube
still round-trips host-side for the (host) softmax — the *online/tiled* FA-2 variant that avoids that is
a dispatch-bound **loss** here as on the LLM path (the host score bandwidth it saves is not the
bottleneck; [attention-offload-crossover.md](../perf/attention-offload-crossover.md)).

**Shipped — bit-exact fp16 GELU LUT.** fp16 GELU is a function of a 16-bit value, so all 65536 outputs
fit one 128 KB table (`g_gelu_lut`, built once over every fp16 bit pattern incl. inf/nan). The host GELU
then turns its per-element `tanhf` into a load — **bit-identical** to the scalar path (it tabulates the
exact same `gelu_pytorch_tanh`), so cosine is unchanged. With attention multicored, GELU had become the
#2 cost (~16 %); the LUT is **1.22× warm** on the whole resident encode (`ROCKET_SIGLIP_GELU_SCALAR=1`
reverts). **Both levers together: 1.78× warm** (4.10 → 2.30 s, contended box; cosine 0.999998) — the
resident encode now runs **under the prior 2.71 s idle baseline even under contention**. [HW sweep,
600 MHz]

Remaining (deferred, lib-level):

1. **Cross-op cube chaining of the resident FFN — measured NET LOSS, do not pursue.** The encoder is
   the regime where cube-chaining pays on the **simple** (single-fd) path — `rocket_mlp_fp16_fused` in
   `rocket_encoder_block_fp16` gives `packA` −20 %, `read` −24 %, total `pack` −30 % there
   ([cross-op-chaining.md](cross-op-chaining.md), [ffn-block.md](ffn-block.md)). But on the **resident**
   path the fc1/fc2 are prepacked **multicore** (3 cores); a cube-resident `fc1 → +bf1 → GELU → fc2`
   chain is inherently **single-fd** (the consumer matmul reads the producer's one cube BO, one IOVA),
   so it forfeits that 3-core parallelism. A prepacked cube-fused MLP (weights packed once, bias
   pre-scattered once) is **~15 % slower** than the multicore host-handoff FFN — the
   lost multicore outweighs the saved host GELU + intermediate de-tile/re-tile, cosine still 0.999987
   [HW sweep 2026-06-29]. So cube-chaining and multicore are mutually exclusive, and on the resident
   path multicore wins. (`LN → matmul` chaining would hit the same single-fd wall.)
2. **GELU** — was the #2 cost; **shipped** the bit-exact fp16 LUT above (1.22×). Moving it on-chip would
   need the cube residency that loses multicore (item 1), and an isolated NPU activation of the
   `[1024,3072]` intermediate is a pack/readback round-trip — so the LUT (a load, no round-trip, bit-exact)
   is the right host lever. Now near-free.
3. **int8-Hadamard** — the quant axis (resident int8 + Hadamard exist); buys RAM, and on this
   readback-bound workload the int32 readback floor likely makes it ~fp16 speed (the LLM finding,
   [k-accumulation.md](k-accumulation.md)) — a fidelity/RAM row, not a latency win.

## Reproduce

```
# on the RK1 (venv with torch+transformers; model + artifacts on /mnt/nvdata):
python tools/siglip_reference.py --out /mnt/nvdata/siglip/artifacts          # fp32 oracle
python tools/siglip_extract.py   --out /mnt/nvdata/siglip/artifacts/siglip_weights.f16
ctest --test-dir build_nv -R siglip_rocket                                   # fidelity gate
sudo rocket-userspace/tools/npu_perf_governor.sh performance                 # REQUIRED for representative latency
ROCKET_KACC=1 ROCKET_SIGLIP_BENCH=12 taskset 0xf0 ./build_nv/siglip_rocket   # +resident bench
#   ROCKET_SIGLIP_PROF=1 adds the per-phase breakdown
sudo rocket-userspace/tools/npu_perf_governor.sh schedutil                   # restore when done
```

The weight blob is a 16×int32 header (magic `SGLP`, dims, eps) + the fp16 weights in declaration
order (patch_W, patch_b, pos, per-layer {ln1, q/k/v/o, ln2, fc1, fc2}, post-LN); the C loader
mmaps it and walks the cursor. All linear weights stay row-major `[out,in]` (PyTorch nn.Linear ==
the matmul's `B=[N,K]`), so no transpose.
