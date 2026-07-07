# Matmul as a 1×1 convolution

The RK3588 NPU has no matmul primitive. It is a convolution engine. So we compute

```
C[M, N] = A[M, K] · B[N, K]ᵀ      (each output channel n = dot(input row m, weight row n))
```

as a **1×1 (pointwise) convolution**:

- the **K** contraction axis becomes the convolution's input channels,
- the **N** output axis becomes the output channels (each weight row n is a
  1×1×K filter),
- the **M** rows become the spatial positions of a 1×K×1 "image."

This is the same CNA→CORE→DPU datapath Mesa drives for real convolutions, which is
why `rocket` accepts the regcmd: from the hardware's point of view it *is* a
convolution.

## The data flow per tile

For one output tile (Mt rows × Nt channels, contracting Kt):

1. **CNA feature load** reads the input-feature tile `A[Mt, Kt]` from DRAM into CBUF,
   via line/surface strides (no reorder — see below).
2. **CNA weight load** reads the weight tile `B[Nt, Kt]` from DRAM into CBUF.
3. **CORE** runs the MAC array: the conv reduces over the Kt input channels in one
   pass (the native K-reduction), accumulating in the wide CACC accumulator.
4. **DPU** writes the output tile `C[Mt, Nt]` back to DRAM in the output cube layout.
   The DPU-RDMA / eltwise sub-unit is involved here (and is the source of the MRDMA
   trap, and the optional fp16 K-accumulate — see those docs).

## The host must pre-scatter the layouts (no on-chip conversion)

The single most important constraint: **the CNA feature/weight load is
stride-addressed only — there is no on-chip row-major→tiled conversion.** [source-confirmed: Mesa `rkt_coefs.c` / `rkt_ml.c`]

- Mesa pre-scatters **both** weights (`rkt_coefs.c`, nested oc1/ic1/x/y/oc2/ic2
  reorder, WEIGHT_ATOMIC_SIZE=32) and input features (`rkt_ml.c`,
  FEATURE_ATOMIC_SIZE=16) into native tiled layout in DRAM **before** upload.
- No register enables a layout conversion: DCOMP disabled, CSC unset, CVT bypassed
  (`CVT_BYPASS=1`), `DATA_FORMAT=0`. The CNA reads DRAM via line/surf stride into
  CBUF banks — a strided read, no reorder. There *is* a weight **decompression**
  engine (`RKNN_cna_dcomp_ctrl`), but it decompresses a compressed stream; it does
  not reorder layout.

**Consequence:** the host-side scatter into native tiles ("packB" for weights,
"packA" for activations) is an **irreducible** cost on this hardware. You can move it
to the fast cores, vectorize the fp32→fp16 convert part, or make tiled weights
resident so you pay it once — but you cannot make the NPU do it. See
[encodings/tile-layouts.md](encodings/tile-layouts.md) for the exact layouts.

## Tiling

A single tile must fit the 12×32 KB CBUF (input tile + weight tile both resident) —
and for **int8** the feature must be given **one bank of slack** beyond `ceil(bytes/bank)`
or its DMA over-reads and garbles the tail rows (see
[encodings/cbuf-bank-slack.md](encodings/cbuf-bank-slack.md)). So large matmuls are tiled
three ways:

- **M (rows)** and **N (output channels)** split into *independent* output blocks —
  each is a separate NPU job, trivially parallel (and the axis we fan across cores).
- **K (contraction)** is split only when Kt would overflow the CBUF. K-partials are
  then summed — on the host in fp32 (the precise default), or on the NPU via the DPU
  eltwise unit for fp16 (the +19% K-accum optimization). See
  [encodings/k-accumulation.md](encodings/k-accumulation.md).

The K-tile count `nKt = ceil(K / Kt)` is the readback multiplier: with host K-accum
you read every output tile `nKt` times (`read ∝ M·N·nKt`); with on-chip K-accum (or
single-pass `nKt=1`) you read it once (`read ∝ M·N`).

### The native K-reduction goes far past one tile

The conv reduces over K in a single pass up to the CBUF limit — HW-tested correct to **K = 10240 in one pass** (fp16). [HW sweep] K tiles to a few hundred only because the output tile (Mt×Nt)
pins how much CBUF is left for Kt. Shrinking Mt/Nt grows Kt and collapses nKt — int4
on Gemma's `K=3840` reaches `nKt=1` (single-pass, zero readback K-accum) at
Mt=Nt=64. (This does not change wall time — the readback is not the binding constraint;
see [perf/not-mac-bound.md](perf/not-mac-bound.md). But it is the right mental model.)

## Alignment requirements

These fall out of the tile layouts (the atomic blocks):

| dtype | K must divide | N must divide | M |
|---|---:|---:|---|
| fp16 | 32 | 16 | %4 (M==1 SW-padded) |
| int8 | 32 | 32 | %4 (M==1 SW-padded) |
| int4 | 32 | 64 | %4 (M==1 SW-padded) |

(bf16 follows fp16 — K%32, N%16; tf32 is K%16, N%16 — a 4-byte element halves the
K-group; int16-exact follows int8 — K%32, N%32.)

N-alignment grows as the weight N-group grows (fp16 16 → int8 32 → int4 64). Getting
N-alignment wrong is silent garbage: e.g. int8 with `N=16` over-reserves a 32-kernel
group and the NPU, reading 16 kernels, disagrees. [HW sweep]

### Feature height < 4 is broken — the M==1 GEMV trap

The M rows are the conv's spatial **height**, and the datapath **produces wrong
output when that height is below 4** [HW-confirmed 2026-06-21]. At height 1 (a
single-vector / GEMV matmul) the result is uncorrelated with the reference
(cosine ≈ 0.01–0.06), across **every** dtype (fp16/int8/int4/int16/bf16/tf32 — they
share the `gen_matmul_*` height geometry). At height ≥ 4 (`M%4==0`) all dtypes are
bit-exact.

This is **distinct** from — and not cured by — the `surf_stride < 0` clamp that fixes
*conv* tiles with < 4 input rows (see [encodings/size-e-quirk.md](encodings/size-e-quirk.md)):
that clamp is present and correct, yet the matmul still mis-computes at height 1, so the
break is a deeper height-< 4 geometry constraint (likely the CACC/CORE atomic
expects a ≥ 4-row block), not just the surface stride.

**Test a matmul with a dtype-agnostic cosine metric on realistic inputs, not integer
inputs** (`tests/matmul_correctness_matrix_rocket.c`): small-integer test inputs give
exact fp16 products that mask this break, and nothing real drives a height-1 matmul to
expose it — LLM decode (the only M==1 case) is GEMV-bound and runs on the CPU (~82× slower
on the NPU — see [perf/not-mac-bound.md](perf/not-mac-bound.md)), and the `ggml-rocket`
backend gates NPU matmul at `ROCKET_MIN_M=4`.

**Workaround (software):** the library's one-shot entry points pad M==1 up to a
height-4 tile (3 zero rows, which contribute 0 — no saturation), compute, and return
row 0. The pure planners and the resident/streaming paths instead require `M%4==0`
and reject M==1 (they cannot cheaply pad pre-packed weights) — so a single-vector
matmul on those paths must be padded to 4 caller-side. `M%4==0` is the real hardware
constraint; `M==1` "works" only because software pads it.
