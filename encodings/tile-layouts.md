# Native tile layouts (CNA feature and weight, DPU output)

The NPU reads and writes data in NVDLA-style tiled "cube" layouts with an atomic channel
block (here called C2). No register converts a layout, so the host scatters the weights
into their tile layout in DRAM before submitting. A matmul's activation and output can
skip theirs: a stride-only row-major form reads A and writes C as plain rows
([../matmul-as-conv.md](../matmul-as-conv.md) §"Layouts"). The layouts below are what the
cube program reads and writes.

A hardware sweep confirms all of them: a wrong layout gives sentinel or garbage columns.
Mesa `rkt_coefs.c`/`rkt_ml.c` cross-checks them [HW sweep, source-confirmed].

## The A / B / C layout matrix

The feature (A), weight (B) and output (C) layouts per datatype, established by hardware
sweep (a wrong layout gives sentinel or garbage columns):

| datatype | A (feature) | B (weight) | C (output) |
|---|---|---|---|
| int4 | `[K/32,M,32]` | `[N/64,K/32,64,32]` | int16 `[N/8,M,8]` |
| int8 | `[K/16,M,16]` | `[N/32,K/32,32,32]` | int32 `[N/4,M,4]` |
| fp16 | `[K/8,M,8]` | `[N/16,K/32,16,32]` | fp32 `[N/4,M,4]` |

The fp16 weight K-group is 32 (against the 4-byte tf32 K-group of 16), which matches the
K>=64 hardware sweep. The RK3588 matmul is same-A/B-datatype only. RKNN's menu has no
mixed-type or int16 pairing, though the silicon runs int16 to int32
([output-transpose-int16.md](output-transpose-int16.md)). The `nt`/packB weight
orientation (`[N,K]` rather than `[K,N]`) matches this project's packB result.

## Feature cube channel atom (C2)

The input feature `A[M, K]` packs as `(K/C2, M, C2)`: K splits into groups of C2, with C2
contiguous innermost. C2 grows as the element gets smaller (denser):

| datatype | feature C2 |
|---|---:|
| fp16 | 8 |
| int8 | 16 |
| int4 | 32 |
| int16 | 8 (== fp16) |
| bf16 | 8 (== fp16, 2-byte) |
| tf32 | 4 (the 16-byte CBUF atom / 4 bytes, the first 4-byte input) |

The fp16 atomic K block in the weight path is 16, and the feature atom is 8: NVDLA's
FEATURE_ATOMIC_SIZE and the weight grouping differ. The C2 atom is `16 bytes / element
size`:

| datatype | element size | C2 atom |
|---|---:|---:|
| fp16, bf16, int16 | 2 B | 8 |
| int8 | 1 B | 16 |
| int4 | ½ B | 32 |
| tf32 | 4 B | 4 |

## Weight layout per datatype

The weight `B[N, K]` reorders by datatype. The N-group is the key difference, and the
source of the N-alignment requirement:

| datatype | weight layout | N-group | K-group | notes |
|---|---|---:|---:|---|
| fp16 | `(N/16, K/32, 16, 32)` | 16 | 32 | `weight_fp16` (code: `(k-1)%16)*32`, K-group 32) |
| int8 | `(N/32, K/32, 32, 32)` | 32 | 32 | `weight_int8` |
| int4 | `(N/64, K/32, 64, 32)` | **64** | 32 | `weight_int4`, nibble-packed |
| int16 | `(N/16, K/32, 16, 32)` | 16 | 32 | `weight_int16` (== `weight_fp16`: int16 and fp16 share the 16-kernel weight group) |
| bf16 | `(N/16, K/32, 16, 32)` | 16 | 32 | `weight_tf32`'s 2-byte sibling, reusing `wt_idx_i16` (== `weight_fp16`) |
| tf32 | `(N/16, K/16, 16, 16)` | 16 | **16** | `weight_tf32`: 4-byte halves the K-group to 16 (N-group stays 16). Still a 1024-byte tile (16·16·4) |

int4 weights are nibble-packed and int8 weights are byte-packed. int4 is 2 values per
byte with HILO=0, and the nibble order is already correct, with no swap. Weight bytes:

| datatype | weight bytes |
|---|---|
| fp16, bf16, int16 | `2·N·K` |
| int8 | `N·K` |
| int4 | `N·K/2` |
| tf32 | `4·N·K` |

### Single-K-group trap

The int4 N-group of 64 is a single-K-group trap. An int4 weight packed with int8's
N-group of 32 coincides with the correct `(N/64, K/32, 64, 32)` layout only at K=32 (one
K-group). So a K=32 test passes and K>32 fails.

tf32 has the same trap from the other direction. For a 4-byte element the weight K-group
is 16, not the fp16/int16 K-group of 32. At a single-K-group shape (K=32 for a K-group of
32, K=16 for a K-group of 16) the weight index is row-major for any grouping. So a K=32
test cannot distinguish K-group 16 from 32: the wrong `(N/8, K/32, 8, 32)` and the correct
`(N/16, K/16, 16, 16)` produce the same bytes there. A K>=64 sweep (plus a K=48 tile,
%16-not-%32) separates them and confirms `(N/16, K/16, 16, 16)`.

**For both int4 and tf32, never reverse-engineer a weight tile at a single-K-group shape.
Test at K >= 2x the candidate K-group.**

## Output cube channel atom (C2)

The DPU writes `C[M, N]` in `(N/C2, M, C2)`. The output element size alone sets the output
C2 (`C2 = 16 bytes / sizeof(out elem)`), whichever input datatype produced it
([precision-field.md](precision-field.md)):

| matmul | output datatype | out elem | output cube C2 |
|---|---|---:|---:|
| fp16×fp16 (default, `fp32tofp16=1`) | fp16 (narrowed) | 2 B | 8 |
| fp16×fp16 (`fp32tofp16=0`) | fp32 | 4 B | 4 |
| int8×int8 | int32 | 4 B | 4 |
| int4×int4 | int16 | 2 B | 8 |
| int16×int16 | int32, saturating (see below) | 4 B | 4 |
| bf16×bf16 | fp32 | 4 B | 4 |
| tf32×tf32 | fp32 | 4 B | 4 |

The rule is C2 = 16 / out-elem-bytes, set by the output element size alone. C2=8 for a
2-byte output (fp16-narrowed matmul, int4->int16), and C2=4 for a 4-byte output (fp32 /
int32). There is no separate "fp16-path fp32 cube". When `gen_matmul_fp16` emits the full
fp32 accumulator (`fp32tofp16=0`, `size_e=3`, `surf×4`), it writes the same C2=4
`out_idx_i16` cube that int8, bf16 and tf32 use. `rocket_matmul_fp16_f32out` proves it on
hardware: reading the fp16-input fp32 output as C2=4 matches an fp64 reference to ~1e-7.

Do not conflate the two fp16-path outputs. The fp16-narrowed 2-byte output is C2=8, and
the fp32 accumulator output is C2=4. The fp32 output cube is C2=4 for every input
datatype. Only the input cube C2 differs by input datatype.

The output stride has its own quirk for the int8 and int4 outputs (`size_e=7` for both),
which int16's int32 output does not share. See [size-e-quirk.md](size-e-quirk.md).

### int16 output

int16's int32 output is the same `[N/4,M,4]` cube as fp16's fp32, written with `size_e` 3,
and it **saturates** to int32. The transposed writer (`tp_org_en`) is the one int16 output
that is not a cube: 8- or 16-bit elements at
`slot = 4m + (na%4) + (na/4)·4M + (n%4)·16M` (`na=n/4`), verified at N<=32. See
[output-transpose-int16.md](output-transpose-int16.md). For an int64-exact matmul over
full-range operands, where int32 saturates, the byte decomposition avoids the native
output.

### int16 through int8 byte decomposition

`rocket_matmul_int16_exact` splits each int16 into two signed bytes with balanced
round-to-nearest: `lo=((x+128)&0xFF)-128`, `hi=(x-lo)>>8`, both in [-128,127]. It runs
four proven int8×int8->int32 matmuls and recombines in int64:
`C = 65536·(Ah·Bh) + 256·(Ah·Bl + Al·Bh) + Al·Bl`. The result is bit-exact with no
saturation, at ~4x the int8 cost.

The domain has one limit. Two signed bytes span [-32896, 32639], so the top 128 int16
codes (32640..32767) are excluded. Full range would need unsigned-low-byte and
sign-correction matmuls. The decomposition is bit-exact on hardware to 2M elements
(`512×3840×4096`, max_abs_err=0).

### int4 output saturation and the Kt cap

int4's int16 output saturates per K-tile, and that sets the in-model Kt cap. The native
int4 matmul reads each K-tile partial back as int16. So any tile whose `|Σ qA·qB|` exceeds
32767 saturates silently (lossy, unrecoverable on the host). For symmetric `[-7,7]` int4 a
partial grows at <= `49·Kt`, so any `Kt > 668` can overflow.

The tiling plan sizes Kt from CBUF alone and can pick the whole K (single-pass). **An
in-model caller that feeds real `[-7,7]` quantized data must cap Kt.**
`rocket_matmul_int4_ex(…, kt_cap)` caps it (the W4A4 LLM path uses 480,
`49·480 < 32767`). `rocket_matmul_int4_groupwise` forces `Kt = group` (<=128, the
per-K-group quant slice).

The bit-exact int4 gates avoid the overflow only by using `[-2,2]` data (`4·3840 < 32767`,
single-pass safe). A real LLM's quantized activations span the full `[-7,7]`. With the
cap, Gemma-4-12B int4 generates char-identically to fp16, and uncapped large-K int4
saturates [HW sweep].

## Consequence for quantization

The denser feature C2 and weight packing are why int4 fits ~4x the K per CBUF bank. That
density lets int4 reach single-pass K (`nKt=1`) where fp16 needs many K-passes. Single-pass
K looks like a big readback win, but in practice the matmul is not readback-bound either
(see [../perf/not-mac-bound.md](../perf/not-mac-bound.md)). So the layouts pay off in RAM
and model size, not in speed.

## `data_entries` divisor

`data_entries` is a related per-datatype constant in the descriptor. A "data entry" is a
fixed 64 bytes, so the divisor is `64 / element-bytes`:

| datatype | element size | divisor |
|---|---:|---:|
| fp16, bf16, int16 | 2 B | 32 |
| int8 | 1 B | 64 |
| int4 | ½ B | 128 |
| tf32 | 4 B | 16 |

The divisor equals the number of K-groups only for the datatypes whose K-group spans a
full 64-byte entry (fp16, bf16, int16, tf32). For int8 and int4 a K-group is smaller than
64 B, so `data_entries` is fewer than the K-group count. The driver has diagnostic knobs
`ROCKET_*_DENTRIES_DIV`.
