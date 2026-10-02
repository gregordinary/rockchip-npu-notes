# Matmul as a 1×1 convolution

The RK3588 NPU has no matmul primitive. It is a convolution engine. So `librocketnpu`
computes

```
C[M, N] = A[M, K] · B[N, K]ᵀ      (each output channel n = dot(input row m, weight row n))
```

as a 1×1 (pointwise) convolution:

- The K contraction axis becomes the convolution's input channels.
- The N output axis becomes the output channels (each weight row n is a K×1×1 filter).
- The M rows become the spatial positions of a K×M×1 "image" (K channels, height M, width 1).

This is the same CNA→CORE→DPU datapath Mesa drives for real convolutions. That is why
`rocket` accepts the regcmd: to the hardware it is a convolution.

## The data flow per tile

For one output tile (Mt rows × Nt channels, contracting Kt):

1. **CNA feature load** reads the input-feature tile `A[Mt, Kt]` from DRAM into CBUF,
   via line and surface strides. It does not reorder (see §"Layouts").
2. **CNA weight load** reads the weight tile `B[Nt, Kt]` from DRAM into CBUF.
3. **CORE** runs the MAC array. The conv reduces over the Kt input channels in one
   pass (the native K-reduction), accumulating in the wide CACC accumulator.
4. **DPU** writes the output tile `C[Mt, Nt]` back to DRAM in the output cube layout.
   The DPU-RDMA / eltwise sub-unit is involved here. It is the source of the MRDMA trap
   and of the optional fp16 K-accumulation (see
   [encodings/mrdma-trap.md](encodings/mrdma-trap.md) and
   [encodings/k-accumulation.md](encodings/k-accumulation.md)).

## Layouts

**No register converts a layout** [source-confirmed: Mesa `rkt_coefs.c` / `rkt_ml.c`, and
the TRM's CNA chapter]. The CNA feature and weight loads are stride-addressed. There is a
weight decompression engine (`RKNN_cna_dcomp_ctrl`), but it decompresses a compressed
stream and does not reorder one. Mesa pre-scatters both weights (`rkt_coefs.c`, nested
oc1/ic1/x/y/oc2/ic2 reorder, WEIGHT_ATOMIC_SIZE=32) and input features (`rkt_ml.c`,
FEATURE_ATOMIC_SIZE=16) into the cube layouts in DRAM before upload.

The weights have to be packed on the host. The weight tile layout
([encodings/tile-layouts.md](encodings/tile-layouts.md)) is the only one the weight load
reads. So a streamed weight pays a host scatter per call, and a resident weight pays it
once.

### Row-major activation and output

For a matmul, the activation and the output need no packing. Strides alone make one task
read A as plain `[M][K]` rows and write C as plain `[M][N]` rows. That takes no scatter of
A and no de-scatter of C on the host. Six registers change against the cube program:

| register | cube program | row-major form |
|---|---|---|
| CNA `CONV_CON1` bit 29 (`GROUP_LINE_OFF`) | 0 | 1 |
| CNA `DMA_CON1` line stride | 4 | K/8, one row of A |
| CNA `DMA_CON2` surface stride | per the cube | 0 |
| DPU `DST_SURF_STRIDE` | M atoms | 1 atom (16 bytes) |
| DPU `DATA_CUBE_NOTCH` (both halves) | 0 | N/4 − 1 |
| DPU `SURFACE_ADD` | 4·M atoms | 4 atoms |

The form is bit-exact for fp16 in and fp32 out, in one task, at 12 shapes with M, K and N
all different. K runs from 64 to 4096 and N from 32 to 2048. The operands are integers,
every element is scored, and the output is prefilled with NaN [HW sweep, RK3588, `rocket`
1.3.0, 600 MHz, 2026-09-26, `tests/rowmajor_matmul_probe`].

The independent generators from allbilly emit the same form. It is exact under the vendor
`rknpu` driver, in fp16 up to K 4096 and in int16 [HW sweep, vendor kernel, 2026-09-26].
Their fp16 generator caps one task at 13 line-stride groups and 384 input channels
(`RK_LINE_STRIDE_GROUP_CAP`, `GEMM_MAX_ALIGN_IN`), and issues one task per row past them.
With both caps lifted it is exact to K 4096. So the caps are the envelope its author
validated, and not a bound of the hardware.

The TRM describes `GROUP_LINE_OFF` as a line-fetch efficiency setting. The allbilly project
reports that it changes the result. No arm here ran the form without it.

### Tiles, int8 and fp16 out

The same registers address a tile inside a larger matrix, which is the task a tiled path
issues. The line stride is one row of the full A, and the feature address is the tile's
first element. A task then reads a K slice out of a wider row. The notch is one row of the
full C, and the destination is the tile. The task then writes an N tile into a wider C and
nothing outside it.

`SURFACE_ADD` is one output group's bytes in atoms. A group is 16 channels for fp16 in and
32 for int8 in. That gives 4 at fp32 out, 2 at fp16 out (notch N/8 − 1), and 8 at int8's
int32 out. Both notch halves (`NOTCH_ADDR_0`, `NOTCH_ADDR_1`) carry the row step. **With
either alone, the task leaves most of the tile unwritten and writes outside it.**

The form is exact on 17 arms, with every element scored over a sentinel-filled output:

- K slices
- N tiles at a row offset
- Both at once
- int8 -> int32, plain and tiled
- fp16 out, plain and tiled
- A two-task fp16 K-accumulation, to 256×768×256. Its partial stays a cube, while the final
  task reads it through the DPU_RDMA and writes rows into a wider C.

The cube program is the control at each precision [HW sweep, RK3588, `rocket` 1.3.0,
600 MHz, 2026-09-26, `tests/rowmajor_tile_probe`]. Under `SURFACE_ADD` 4 and a notch sized
for four bytes a channel, an fp16 output lands in runs of 16 channels with 16-channel gaps.
That is the layout the fp16 generator from allbilly reads back.

### Device cost

The device cost is the output's row pitch. At the library's tile shapes (256×384×256 and
256×512×128 fp16 -> fp32, 256×384×256 fp16 -> fp16, 256×512×256 int8 -> int32), a row-major
task at the tile's own width runs 0.96-1.12x the cube program's submit-to-fence.

The row-major read of A costs 1.00-1.07x at 256×384×256 and 1.06-1.21x at 256×512×128. The
second is the shape with the least MAC work for each byte of A it reads. At neither shape
does the read cost grow with the pitch, from the tile's own width to 16 KiB a row. Writing
into a wider C costs time that does grow with the pitch once a row passes 4 KiB. At
256×384×256 fp16 -> fp32:

| C row pitch | 1 KiB (own) | 2 KiB | 4 KiB | 8 KiB | 16 KiB | 32 KiB |
|---|---|---|---|---|---|---|
| submit-to-fence, as a multiple of the cube program | 0.88-0.95 | 0.99-1.00 | 0.85-1.07 | 1.12-1.25 | 1.43-1.60 | 2.01-2.29 |

Each cell is the range over processes of the median per-rep ratio, at 40 rotated reps a
process. Two processes ran at 1 and 2 KiB, and four at the rest.

The 256×512×128 shape has half the output channel groups. There the same sweep reads
1.10-1.13x at 8 KiB, 1.29-1.38x at 16 KiB and 1.61-1.74x at 32 KiB, about half the added
time. So the cost goes with the rows written times the pitch. A pitch 16 bytes off a power
of two (2052, 4100, 4160 and 8196 elements) costs what its power-of-two neighbor does. So
the cost is not DRAM bank aliasing [HW sweep, same].

That the write DMA steps through the notch at a fixed rate is [hypothesis]. The CNA reads
at the same pitches, through the same IOMMU, at no cost.

Each wait was on a 4 KiB BO that the job lists as an output, and every arm's BOs were the
same size. The reason is that `PREP_BO` syncs the whole BO it waits on
([perf/bo-sync-cost.md](perf/bo-sync-cost.md)). Waiting on a 4 MiB row-major C against the
cube program's 256 KiB, the four shapes read 1.36-2.07x. The same programs read 1.22-1.73x
with the wait moved to the small BO.

At a model's width, a row of C is 4-60 KiB. So a task that writes straight into C is slower
on the device than the cube program. The form that costs the device nothing is a row-major
tile at its own width, which the host then places as whole rows.

### Cost in a tiled fp16 job

In the job a tiled fp16 path issues, the form costs 0-7%. That job is one chained kick,
with K slices outer and N tiles inner. Each slice after the first adds the previous fp16
partial through the DPU_RDMA. `DATA_REUSE` keeps a slice of A in the CBUF across the N
tiles. The row-major read works under reuse: every arm is exact at three shapes. The table
gives the submit-to-fence medians over 100 rotated reps, against the cube program
[HW sweep, RK3588, `rocket` 1.3.0, 600 MHz, 2026-09-26, `tests/rowmajor_kacc_chain_probe`]:

| Shape (tiles), C row | Own-width tiles | Straight into C | Own-width, reuse off |
|---|---|---|---|
| 256×1536×1024 (384×256), 2 KiB | 1.07x | 1.06x | 1.17x |
| 256×2048×2048 (512×128), 4 KiB | 1.00x | 1.00x | 1.22x |
| 256×1536×4096 (384×256), 8 KiB | 1.02x | 1.17x | 1.15x |

The per-rep ratios are bimodal by position in the rotation. So the figure is the median of
each arm's own times, and each arm ran at every position equally often.

### Scope

These are not measured:

- A multi-core or resident job
- int4, bf16 or tf32, and int16 under `rocket`
- A K-accumulation partial kept row-major
- What the removed host work is worth

The host work has one profile, of a fully resident 12B F16 model at pp2048. There the A
pack and the output read are together at most 5.3% of the prefill wall [expected, a bound
read from one profile]. A device add of 0-7% leaves that no margin, so `librocketnpu` still
packs A and de-scatters C.

## Tiling

A single tile must fit the 12×32 KB CBUF (input tile + weight tile both resident). **For
int8, the feature must be given one bank of slack beyond `ceil(bytes/bank)`.** Without it,
the feature DMA over-reads and garbles the tail rows (see
[encodings/cbuf-bank-slack.md](encodings/cbuf-bank-slack.md)). So large matmuls are tiled
three ways:

- **M (rows)** and **N (output channels)** split into independent output blocks. Each is a
  separate NPU job, trivially parallel, and this is the axis `librocketnpu` fans across
  cores.
- **K (contraction)** splits only when Kt would overflow the CBUF. The K-partials are
  then summed on the host in fp32 (the byte-exact fallback), or on the NPU via the DPU
  eltwise unit for fp16 (the default, +19%). See
  [encodings/k-accumulation.md](encodings/k-accumulation.md).

The K-tile count `nKt = ceil(K / Kt)` is the readback multiplier. With host K-accumulation
you read every output tile `nKt` times (`read ∝ M·N·nKt`). With on-chip K-accumulation (or
single-pass `nKt=1`) you read it once (`read ∝ M·N`).

### Per-tile geometry limits

A task's extents sit in fixed-width fields, and for a convolution the CNA's are the
narrow ones. The CNA holds the input width and height and the output width in 11 bits
each [TRM: `RKNN_cna_data_size0` bits 26:16 and 10:0, `RKNN_cna_data_size2` bits 10:0].
`FEATURE_GRAINS`, which a generator sets to the input height plus one, holds 10 bits
[source-confirmed: Mesa `registers.xml`]. So one task sees at most 1022 input rows by 2047
columns. The DPU's cube fields hold 13 bits, so they do not bind a convolution.

A value past its field wraps, and the task computes a full surface of the wrapped extent.
An int8 conv tile 2500 columns wide programs as 452. The task writes the first 452 columns
right and the other 2048 wrong, and the call returns success [HW sweep, RK3588, `rocket`
1.3.0, 2026-09-26, `tests/conv_width_gate`]. At 32 input channels a 2×2300 plane comes back
wrong on 147199 of 147200 elements.

A CBUF budget bounds a tile's area and not its axes. So a thin-channel convolution, 32
channels or fewer after padding, with one short axis can pass the limit on the other. A
tiler must cap each axis separately.

Two other guards in `librocketnpu` confine the wrap to that one case. The `FEATURE_GRAINS`
check refuses a tall plane before it wraps. The bank check refuses an fp16 plane before it
reaches a width of 2048.

A short tile also costs more CBUF than its rows suggest. A job pads a sub-input below four
rows up to four (see §"Feature height floor"). So a budget taken at the real rows admits a
tile that the bank check then refuses.

### Native K-reduction

The conv reduces over K in a single pass up to the CBUF limit. It is hardware-tested
correct to K = 10240 through the tiler (fp16) [HW sweep]. K tiles to a few hundred only because
the output tile (Mt×Nt) fixes how much CBUF is left for Kt. Shrinking Mt and Nt grows Kt
and collapses nKt. int4 on Gemma's `K=3840` reaches `nKt=1` (single-pass, zero readback
K-accumulation) at Mt=Nt=64.

This does not change wall time, because the readback is not the binding constraint (see
[perf/not-mac-bound.md](perf/not-mac-bound.md)). It is still the right mental model.

## Alignment requirements

These follow from the tile layouts (the atomic blocks):

| datatype | K must divide | N must divide | M |
|---|---:|---:|---|
| fp16 | 32 | 16 | %4 (M==1 sw-padded) |
| int8 | 32 | 32 | %4 (M==1 sw-padded) |
| int4 | 32 | 64 | %4 (M==1 sw-padded) |

bf16 follows fp16 at K%32, N%16. tf32 is K%16, N%16: a 4-byte element halves the K-group.
int16-exact follows int8 at K%32, N%32.

N-alignment grows as the weight N-group grows (fp16 16 -> int8 32 -> int4 64). **A wrong
N-alignment gives silent garbage** [HW sweep]. For example, int8 with `N=16` over-reserves a
32-kernel group, and the NPU, reading 16 kernels, disagrees.

### Feature height floor (the M==1 GEMV trap)

The M rows are the conv's spatial height, and **the datapath produces wrong output when
that height is below 4** [HW sweep, 2026-06-21]. At height 1 (a single-vector, GEMV matmul)
the result is uncorrelated with the reference (cosine ~0.01-0.06). That holds for every
datatype: fp16, int8, int4, int16, bf16 and tf32 share the `gen_matmul_*` height geometry.
At height >= 4 (`M%4==0`) all datatypes are bit-exact.

This break is distinct from the `surf_stride < 0` clamp, which fixes conv tiles with < 4
input rows (see [encodings/size-e-quirk.md](encodings/size-e-quirk.md)). The clamp does not
cure it. The clamp is present and correct, yet the matmul still mis-computes at height 1.
So the break is a height-< 4 geometry constraint, not only the surface stride. The CACC/CORE
atomic likely expects a >= 4-row block.

**Test a matmul with a datatype-agnostic cosine metric on realistic inputs, not integer
inputs** (`tests/matmul_correctness_matrix_rocket.c`). Small-integer test inputs give exact
fp16 products that mask this break. Nothing real drives a height-1 matmul to expose it. LLM
decode (the only M==1 case) is GEMV-bound and runs on the CPU (~82x slower on the NPU, see
[perf/not-mac-bound.md](perf/not-mac-bound.md)). The `ggml-rocket` backend gates NPU matmul
at `ROCKET_MIN_M=4`.

#### Software workaround

The library's one-shot entry points pad M==1 up to a height-4 tile (3 zero rows, which
contribute 0 and cannot saturate), compute, and return row 0. The pure planners and the
resident and streaming paths instead require `M%4==0` and reject M==1, because they cannot
cheaply pad pre-packed weights. So the caller must pad a single-vector matmul on those
paths to 4 rows. `M%4==0` is the hardware constraint. `M==1` works only because software
pads it.
