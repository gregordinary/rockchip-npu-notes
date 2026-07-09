# Native tile layouts (CNA feature / weight; DPU output)

The NPU reads and writes data in NVDLA-style **tiled "cube" layouts** with an atomic
channel block (here called **C2**). Because there is **no on-chip layout conversion**
(see [../matmul-as-conv.md](../matmul-as-conv.md)), the host must scatter row-major
data into these layouts in DRAM before submitting, and de-scatter the output after.
This host packing is the irreducible per-call cost; the layouts below are exactly
what you must produce.

All confirmed by **HW sweep** (a wrong layout = sentinel/garbage columns) and
cross-checked against Mesa `rkt_coefs.c`/`rkt_ml.c`.

## The A / B / C layout matrix

The full per-dtype feature (A), weight (B) and output (C) layouts, established by HW sweep
(a wrong layout = sentinel/garbage columns):

| dtype | A (feature) | B (weight) | C (output) |
|---|---|---|---|
| int4 | `[K/32,M,32]` | `[N/64,K/32,64,32]` | int16 `[N/8,M,8]` |
| int8 | `[K/16,M,16]` | `[N/32,K/32,32,32]` | int32 `[N/4,M,4]` |
| fp16 | `[K/8,M,8]` | `[N/16,K/32,16,32]` | fp32 `[N/4,M,4]` |

Two points worth keeping: (1) the **fp16 weight K-group is 32** (vs the 4-byte tf32 K-group
of 16), matching the K≥64 HW sweep; and (2) the RK3588 matmul is **same-A/B-dtype only** —
there is no native mixed-type or native int16 path here
([output-transpose-int16.md](output-transpose-int16.md)). The `nt`/packB weight orientation
(`[N,K]` rather than `[K,N]`) matches our packB result.

## Feature / input cube — channel atom C2

The input feature `A[M, K]` is packed `(K/C2, M, C2)` — i.e. K is split into groups
of C2, with C2 contiguous innermost. C2 shrinks as the element gets smaller (denser):

| dtype | feature C2 |
|---|---:|
| fp16 | 8 |
| int8 | 16 |
| int4 | 32 |
| int16 | 8 (== fp16) |
| bf16 | 8 (== fp16, 2-byte) |
| **tf32** | **4** (the 16-byte CBUF atom / 4 bytes — the first 4-byte input) |

(The fp16 atomic K block in the weight path is 16; the feature atom is 8 — NVDLA's
FEATURE_ATOMIC_SIZE vs the weight grouping differ.) The C2 atom is `16 bytes / element
size`: fp16/bf16/int16 (2 B) → 8, int8 (1 B) → 16, int4 (½ B) → 32, **tf32 (4 B) → 4**.

## Weight layout — per dtype

The weight `B[N, K]` reorders by dtype. The N-group is the key difference (and the
source of the N-alignment requirement):

| dtype | weight layout | N-group | K-group | notes |
|---|---|---:|---:|---|
| **fp16** | `(N/16, K/32, 16, 32)` | 16 | 32 | `weight_fp16` (code: `(k-1)%16)*32`, K-group 32) |
| **int8** | `(N/32, K/32, 32, 32)` | 32 | 32 | `weight_int8` |
| **int4** | `(N/64, K/32, 64, 32)` | **64** | 32 | `weight_int4`, nibble-packed |
| **int16** | `(N/16, K/32, 16, 32)` | 16 | 32 | `weight_int16` (== `weight_fp16`; int16 and fp16 share the 16-kernel weight group) |
| **bf16** | `(N/16, K/32, 16, 32)` | 16 | 32 | `weight_tf32`'s 2-byte sibling — reuses `wt_idx_i16` (== `weight_fp16`) |
| **tf32** | `(N/16, K/16, 16, 16)` | 16 | **16** | `weight_tf32` — **4-byte halves the K-group to 16** (N-group stays 16); still a 1024-byte tile (16·16·4) |

The int4 N-group-of-64 is a
single-K-group trap: an int4 weight packed with int8's N-group-of-32 *coincides* with the
correct `(N/64, K/32, 64, 32)` layout only at K=32 (one K-group), so a K=32 test passes and
K>32 fails.

**tf32 has the same trap from the other direction.** For a 4-byte element the weight
K-group is **16**, not the fp16/int16 K-group of 32. At a single-K-group shape (K=32 for a
K-group of 32, K=16 for a K-group of 16) the weight index is row-major for *any* grouping,
so a K=32 test cannot distinguish K-group 16 from 32: the wrong `(N/8, K/32, 8, 32)` and the
correct `(N/16, K/16, 16, 16)` produce the same bytes there. A K≥64 sweep (plus a K=48 tile,
%16-not-%32) separates them and confirms `(N/16, K/16, 16, 16)`. **Rule (both int4 and
tf32): never RE a weight tile at a single-K-group shape; test at K ≥ 2× the candidate
K-group.**

int4/int8 weights are nibble/byte-packed (int4 = 2 values per byte, HILO=0 — the
nibble order is already correct, no swap). Weight bytes: fp16/bf16/int16 = `2·N·K`,
int8 = `N·K`, int4 = `N·K/2`, **tf32 = `4·N·K`**.

## Output cube — channel atom C2 (per output dtype)

The DPU writes `C[M, N]` in `(N/C2, M, C2)` with an output C2 that is set purely by the
**output element SIZE** (`C2 = 16 bytes / sizeof(out elem)`), regardless of which input
dtype produced it — see [precision-field.md](precision-field.md):

| matmul | output dtype | out elem | output cube C2 |
|---|---|---:|---:|
| fp16×fp16 (default, `fp32tofp16=1`) | fp16 (narrowed) | 2 B | 8 |
| **fp16×fp16 (`fp32tofp16=0`)** | **fp32** | **4 B** | **4** |
| int8×int8 | int32 | 4 B | 4 |
| int4×int4 | int16 | 2 B | 8 |
| int16×int16 | (no native int32 output — see below) | — | — |
| bf16×bf16 | fp32 | 4 B | 4 |
| tf32×tf32 | fp32 | 4 B | 4 |

The rule is **C2 = 16 / out-elem-bytes**, set by the output element size alone: C2=8 for a
**2-byte** output (fp16-*narrowed* matmul, int4→int16), C2=4 for a **4-byte** output (fp32 /
int32). There is no separate "fp16-path fp32 cube": when `gen_matmul_fp16` emits the full
fp32 accumulator (`fp32tofp16=0`, `size_e=3`, `surf×4`) it writes the **same C2=4
`out_idx_i16` cube** that int8/bf16/tf32 use. HW-proven by `rocket_matmul_fp16_f32out`:
reading the fp16-input fp32 output as C2=4 matches an fp64 reference to ~1e-7. Don't
conflate the two fp16-path outputs — the fp16-*narrowed* 2-byte output is genuinely C2=8,
the fp32 accumulator output is C2=4. The fp32 output cube is C2=4 for every input dtype;
only the *input* cube C2 differs by input dtype.

### int16 output is a special case — not a simple cube

Unlike the others, **int16 has no native full-iteration int32 output cube.** The DPU
either writes a single int32 tile (broken iteration) or, via `tp_org_en`, a full but
8/16-bit **transposed, saturating** buffer with a non-cube layout
`slot = 4m + (na%4) + (na/4)·4M + (n%4)·16M` (`na=n/4`), verified at N≤32. See
[output-transpose-int16.md](output-transpose-int16.md) for the full story. For a real
full-precision int16 matmul we bypass the native output entirely:

**int16 via int8 byte-decomposition (`rocket_matmul_int16_exact`).** Split each int16
into two signed bytes (balanced round-to-nearest: `lo=((x+128)&0xFF)-128`,
`hi=(x-lo)>>8`, both in [-128,127]) and run **four** proven int8×int8→int32 matmuls,
recombining in int64: `C = 65536·(Ah·Bh) + 256·(Ah·Bl + Al·Bh) + Al·Bl`. Bit-exact,
no saturation, ~4× int8 cost. Domain caveat: two signed bytes span [-32896, 32639],
so the top 128 int16 codes (32640..32767) are excluded (full range would need
unsigned-low-byte + sign-correction matmuls). HW-verified bit-exact to 2M elements
(`512×3840×4096`, max_abs_err=0).

**int4's int16 output saturates per K-tile — the in-model Kt cap.** The native int4
matmul reads each K-tile partial back as int16, so any tile whose `|Σ qA·qB|` exceeds
32767 saturates *silently* (lossy, unrecoverable on the host). For symmetric `[-7,7]`
int4 a partial grows at ≤ `49·Kt`, so any `Kt > 668` can overflow. The tiling plan sizes
Kt from CBUF alone — it can pick the whole K (single-pass) — so an in-model caller that
feeds real `[-7,7]` quantized data **must** cap Kt: `rocket_matmul_int4_ex(…, kt_cap)`
caps it (the W4A4 LLM path uses 480, `49·480 < 32767`), and `rocket_matmul_int4_groupwise`
forces `Kt = group` (≤128, the per-K-group quant slice). The bit-exact int4 gates only
dodged this by using `[-2,2]` data (`4·3840 < 32767`, single-pass safe); a real LLM's
quantized activations span the full `[-7,7]`. [HW sweep — Gemma-4-12B int4 generates
char-identically to fp16 with the cap; uncapped large-K int4 saturates.]

The output **stride** has its own quirk for integer outputs (`size_e=7` for both
int32 and int16) — see [size-e-quirk.md](size-e-quirk.md).

## Why this all matters for quantization

The denser feature C2 and weight packing are exactly why int4 fits ~4× the K per CBUF
bank, which is what lets int4 reach single-pass K (`nKt=1`) where fp16 needs many
K-passes. That *should* be a big readback win — but in practice the matmul is not
readback-bound either (see [../perf/not-mac-bound.md](../perf/not-mac-bound.md)), so
the layouts' real payoff is **RAM / model size**, not speed.

## `data_entries` divisor

A related per-dtype constant in the descriptor: a "data entry" is a fixed **64 bytes**, so
the divisor = `64 / element-bytes`: fp16/bf16/int16 (2 B) → **32**, int8 (1 B) → **64**,
int4 (½ B) → **128**, **tf32 (4 B) → 16**. This equals the number of K-groups only for the dtypes
whose K-group spans a full 64-byte entry (fp16/bf16/int16, tf32); for int8/int4 a K-group is smaller
than 64 B, so `data_entries` is fewer than the K-group count. (Diagnostic knobs
`ROCKET_*_DENTRIES_DIV` exist in the driver.)
