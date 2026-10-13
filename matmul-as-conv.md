# Matmul as a 1×1 convolution

The RK3588 NPU is a convolution engine with no matmul primitive. Software computes a matmul

```
C[M, N] = A[M, K] · B[N, K]ᵀ      (each output channel n = dot(input row m, weight row n))
```

as a 1×1 (pointwise) convolution:

- The K contraction axis becomes the convolution's input channels.
- The N output axis becomes the output channels (each weight row n is a K×1×1 filter).
- The M rows become the spatial positions of a K×M×1 "image" (K channels, height M, width 1).

The program runs on the same CNA→CORE→DPU datapath that Mesa drives for convolutions. `rocket`
accepts the regcmd because, to the hardware, it is a convolution.

## Data flow per tile

For one output tile (Mt rows × Nt channels, contracting Kt):

1. **CNA feature load** reads the input-feature tile `A[Mt, Kt]` from DRAM into CBUF,
   via line and surface strides. It does not reorder (see §"Layouts").
2. **CNA weight load** reads the weight tile `B[Nt, Kt]` from DRAM into CBUF.
3. **CORE** runs the MAC array. The conv reduces over the Kt input channels in one
   pass (the native K-reduction), accumulating in the wide CACC accumulator.
4. **DPU** writes the output tile `C[Mt, Nt]` back to DRAM in the output cube layout.
   The DPU_RDMA and the eltwise sub-unit take part here. They are the source of the MRDMA
   trap and of the optional fp16 K-accumulation (see
   [encodings/mrdma-trap.md](encodings/mrdma-trap.md) and
   [encodings/k-accumulation.md](encodings/k-accumulation.md)).

## Alignment requirements

The alignment rules follow from the atomic blocks of the tile layouts
([encodings/tile-layouts.md](encodings/tile-layouts.md)). Software must keep K, N and M
aligned as the table gives:

| datatype | K must divide | N must divide | M |
|---|---:|---:|---|
| fp16 | 32 | 16 | %4 (M==1 sw-padded) |
| int8 | 32 | 32 | %4 (M==1 sw-padded) |
| int4 | 32 | 64 | %4 (M==1 sw-padded) |

bf16 follows fp16 at K%32, N%16. tf32 is K%16, N%16: a 4-byte element halves the K-group.
int16-exact follows int8 at K%32, N%32.

N-alignment grows as the weight N-group grows (fp16 16 -> int8 32 -> int4 64).

> [!CAUTION]
> A wrong N-alignment produces incorrect output with no error [HW sweep]. For example, int8
> with `N=16` over-reserves a 32-kernel group, and the NPU, reading 16 kernels, disagrees.

### Minimum feature height

> [!CAUTION]
> Erratum E3: a matmul with fewer than 4 rows (M, the convolution's spatial height) produces
> incorrect output [HW sweep 2026-06-21].
> At height 1 (a single-vector, GEMV matmul) the result is uncorrelated with the reference
> (cosine ~0.01-0.06). Software must keep `M%4==0`, and pad a single-row matmul to 4 rows.

The M rows are the conv's spatial height. The limit holds for every datatype: fp16, int8,
int4, int16, bf16 and tf32 share the `gen_matmul_*` height geometry. At height >= 4
(`M%4==0`) all datatypes are bit-exact [HW sweep 2026-06-21].

The `surf_stride < 0` clamp, which fixes conv tiles with < 4 input rows (see
[encodings/size-e-quirk.md](encodings/size-e-quirk.md)), does not cure the limit. The limit is
a height-< 4 geometry constraint, not only the surface stride. The CACC/CORE
atomic likely expects a >= 4-row block [hypothesis].

> [!NOTE]
> Small-integer test inputs give exact fp16 products that mask this limit. Test a matmul with a
> datatype-agnostic cosine metric on realistic inputs, not integer inputs.

No real workload drives a height-1 matmul to expose the limit. LLM decode, the only M==1
case, is GEMV-bound and runs on the CPU (see [perf/decode-gemv.md](perf/decode-gemv.md)).

## Tiling

A single tile must fit the 12×32 KB CBUF, with the input tile and the weight tile both
resident. Large matmuls are therefore tiled three ways:

- **M (rows)** and **N (output channels)** split into independent output blocks. Each is a
  separate NPU job, trivially parallel, and this is the axis `librocketnpu` fans across
  cores.
- **K (contraction)** splits only at a Kt too large for the CBUF. The K-partials are
  then summed on the host in fp32 (the byte-exact fallback), or on the NPU via the DPU
  eltwise unit for fp16 (the default, +19%). See
  [encodings/k-accumulation.md](encodings/k-accumulation.md).

> [!CAUTION]
> For int8, software must give the feature one bank of slack beyond `ceil(bytes/bank)` (known
> issue E4). Without it, the feature DMA over-reads and corrupts the tail rows (see
> [encodings/cbuf-bank-slack.md](encodings/cbuf-bank-slack.md)).

The K-tile count `nKt = ceil(K / Kt)` is the readback multiplier. With host K-accumulation
the host reads every output tile `nKt` times (`read ∝ M·N·nKt`). With on-chip
K-accumulation (or single-pass `nKt=1`) it reads each tile once (`read ∝ M·N`).

### Per-tile geometry limits

A task's extents sit in fixed-width fields, and for a convolution the CNA's are the
narrow ones. The CNA holds the input width and height and the output width in 11 bits
each [TRM: `RKNN_cna_data_size0` bits 26:16 and 10:0, `RKNN_cna_data_size2` bits 10:0].
`FEATURE_GRAINS`, which software sets to the input height plus one, holds 10 bits
[source-confirmed: Mesa `registers.xml`]. One task therefore reads at most 1022 input rows by
2047 columns. The DPU's cube fields are 13 bits wide, and they do not limit a convolution.

> [!CAUTION]
> Erratum E8: a value past its field wraps, and the task writes a complete output of the
> wrapped extent. An int8 conv tile 2500 columns wide programs as 452. The task writes the
> first 452 columns correctly and the other 2048 incorrectly, and the call returns success
> [HW sweep, RK3588, `rocket` 1.3.0, 2026-09-26]. Software must keep each task within 1022
> input rows by 2047 columns.

A CBUF budget bounds a tile's area and not its axes. A thin-channel convolution, 32 channels
or fewer after padding, with one short axis can therefore pass the limit on the other.
Software must cap each axis of a tile separately.

### Native K-reduction

The conv reduces over K in a single pass up to the CBUF limit. Through the tiler, fp16 is
hardware-tested correct to K = 10240 [HW sweep]. K tiles to a few hundred only because the
output tile (Mt×Nt) fixes how much CBUF is left for Kt. Shrinking Mt and Nt grows Kt and
collapses nKt. int4 on Gemma's `K=3840` reaches `nKt=1` (single-pass, zero readback
K-accumulation) at Mt=Nt=64.

Collapsing nKt does not change wall-clock time, because the readback is not the binding
constraint (see [perf/not-mac-bound.md](perf/not-mac-bound.md)).

## Layouts

No register converts a layout [source-confirmed: Mesa `rkt_coefs.c` / `rkt_ml.c`, and
the TRM's CNA chapter]. The CNA feature and weight loads are stride-addressed. There is a
weight decompression engine (`RKNN_cna_dcomp_ctrl`), but it decompresses a compressed
stream and does not reorder one. Mesa pre-scatters both weights (`rkt_coefs.c`, nested
oc1/ic1/x/y/oc2/ic2 reorder, WEIGHT_ATOMIC_SIZE=32) and input features (`rkt_ml.c`,
FEATURE_ATOMIC_SIZE=16) into the cube layouts in DRAM before upload.

Software must pack the weights on the host. The weight tile layout
([encodings/tile-layouts.md](encodings/tile-layouts.md)) is the only one the weight load
reads. A streamed weight therefore costs a host scatter per call, and a resident weight
costs it once.

### Row-major activation and output

**Conditions:** RK3588, `rocket` 1.3.0, 600 MHz.

**Not characterized:** the form in these cases:

- int4, bf16 or tf32, and int16 under `rocket`
- A K-accumulation partial kept row-major
- The general matmul paths with the form
- The form with `GROUP_LINE_OFF` clear

For a matmul, the activation and the output need no packing. With the strides programmed for
it, one task reads A as plain `[M][K]` rows and writes C as plain `[M][N]` rows. The host
then does no scatter of A and no de-tile of C. The form differs from the cube program in
six registers:

| register | cube program | row-major form |
|---|---|---|
| CNA `CONV_CON1` bit 29 (`GROUP_LINE_OFF`) | 0 | 1 |
| CNA `DMA_CON1` line stride | 4 | K/8, one row of A |
| CNA `DMA_CON2` surface stride | per the cube | 0 |
| DPU `DST_SURF_STRIDE` | M atoms | 1 atom (16 bytes) |
| DPU `DATA_CUBE_NOTCH` (both halves) | 0 | N/4 − 1 |
| DPU `SURFACE_ADD` | 4·M atoms | 4 atoms |

The form is bit-exact for fp16 in and fp32 out, in one task. K runs from 64 to 4096 and N
from 32 to 2048 [HW sweep 2026-09-26]. Its device cost depends on the output's row pitch
(§"Cost of the row-major form").

The independent generators from allbilly emit the same form. It is exact under the vendor
`rknpu` driver, in fp16 up to K 4096 and in int16 [HW sweep, vendor kernel, 2026-09-26].
Their fp16 generator caps one task at 13 line-stride groups and 384 input channels
(`RK_LINE_STRIDE_GROUP_CAP`, `GEMM_MAX_ALIGN_IN`), and issues one task per row past them.
With both caps lifted it is exact to K 4096 [HW sweep, vendor kernel, 2026-09-26]. The caps are
therefore the range its author validated, and not a bound of the hardware.

The TRM describes `GROUP_LINE_OFF` as a line-fetch efficiency setting. The allbilly project
reports that it changes the result.

### Tiles, int8 and fp16 out

**Conditions:** RK3588, `rocket` 1.3.0, 600 MHz.

The same registers address a tile inside a larger matrix, which is the task a tiled path
issues. The line stride is one row of the full A, and the feature address is the tile's
first element. A task then reads a K slice out of a wider row. The notch is one row of the
full C, and the destination is the tile. The task then writes an N tile into a wider C and
nothing outside it.

`SURFACE_ADD` is one output group's bytes in atoms. A group is 16 channels for fp16 in and
32 for int8 in. `SURFACE_ADD` is therefore 4 at fp32 out, 2 at fp16 out (notch N/8 − 1), and
8 at int8's int32 out. Software must program the row step in both notch halves
(`NOTCH_ADDR_0`, `NOTCH_ADDR_1`).

> [!CAUTION]
> With the row step in either notch half alone, the task leaves most of the tile unwritten and
> writes outside it.

The form is exact at each of these, with the cube program as the control at each precision
[HW sweep 2026-09-26]:

- K slices
- N tiles at a row offset
- Both at once
- int8 -> int32, plain and tiled
- fp16 out, plain and tiled
- A two-task fp16 K-accumulation, to 256×768×256. Its partial stays a cube, while the final
  task reads it through the DPU_RDMA and writes rows into a wider C.

Under `SURFACE_ADD` 4 and a notch sized for four bytes a channel, an fp16 output is written in
runs of 16 channels with 16-channel gaps. The fp16 generator from allbilly reads back that
layout.

## Cost of the row-major form

The figures below compare the row-major form's submit-to-fence time with the cube program's.

### Device cost

**Conditions:** RK3588, `rocket` 1.3.0, 600 MHz, 2026-09-26. Each wait is on a 4 KiB BO that
the job lists as an output, and every configuration's BOs are the same size.

The device cost depends on the output's row pitch. At the library's tile shapes (256×384×256
and 256×512×128 fp16 -> fp32, 256×384×256 fp16 -> fp16, 256×512×256 int8 -> int32), a
row-major task at the tile's own width runs 0.96-1.12x the cube program's submit-to-fence.

The row-major read of A costs 1.00-1.07x at 256×384×256 and 1.06-1.21x at 256×512×128. The
second is the shape with the least MAC work for each byte of A it reads. At neither shape
does the read cost grow with the pitch, from the tile's own width to 16 KiB a row. Writing
into a wider C costs time that does grow with the pitch once a row passes 4 KiB. At
256×384×256 fp16 -> fp32:

| C row pitch | 1 KiB (own) | 2 KiB | 4 KiB | 8 KiB | 16 KiB | 32 KiB |
|---|---|---|---|---|---|---|
| submit-to-fence, as a multiple of the cube program | 0.88-0.95 | 0.99-1.00 | 0.85-1.07 | 1.12-1.25 | 1.43-1.60 | 2.01-2.29 |

Each value is the range over processes of the median per-rep ratio.

The 256×512×128 shape has half the output channel groups. There the same sweep measures
1.10-1.13x at 8 KiB, 1.29-1.38x at 16 KiB and 1.61-1.74x at 32 KiB, about half the added
time. The cost therefore scales with the rows written times the pitch [HW sweep].

The proposed cause is that the write DMA steps through the notch at a fixed rate
[hypothesis]. The CNA reads at the same pitches, through the same IOMMU, at no cost.

> [!NOTE]
> `PREP_BO` syncs the whole BO it waits on ([perf/bo-sync-cost.md](perf/bo-sync-cost.md)).
> Waiting on a 4 MiB row-major C against the cube program's 256 KiB, the four shapes measure
> 1.36-2.07x. The same programs measure 1.22-1.73x with the wait moved to the small BO.

At a model's width, a row of C is 4-60 KiB. A task that writes straight into C is therefore
slower on the device than the cube program. A row-major tile at its own width costs the device
nothing, and the host then places it as whole rows.

### Cost in a tiled fp16 job

**Conditions:** RK3588, `rocket` 1.3.0, 600 MHz, 2026-09-26.

In the job a tiled fp16 path issues, the form costs 0-7%. The job is one chained kick,
with K slices outer and N tiles inner. Each slice after the first adds the previous fp16
partial through the DPU_RDMA. `DATA_REUSE` keeps a slice of A in the CBUF across the N
tiles. The row-major read works under reuse: every configuration is exact at three shapes.
The table gives the submit-to-fence medians against the cube program
[HW sweep]:

| Shape (tiles), C row | Own-width tiles | Straight into C | Own-width, reuse off |
|---|---|---|---|
| 256×1536×1024 (384×256), 2 KiB | 1.07x | 1.06x | 1.17x |
| 256×2048×2048 (512×128), 4 KiB | 1.00x | 1.00x | 1.22x |
| 256×1536×4096 (384×256), 8 KiB | 1.02x | 1.17x | 1.15x |

### In the flash attention

`librocketnpu`'s flash attention hands the scores and P between its two batched matmuls in this
form. The jobs are multi-task, on resident BOs, one a worker across the cores. QK writes its scores
as plain rows to a 4 KiB row. Past that it writes each N tile as its own block of rows at the
tile's width, which costs the device nothing at any width. AV reads P as plain rows at the full
pitch, which costs nothing measurable at 64 KiB a row in that job.

The form removes the scores' de-tile and P's pack. The call is 0.64-0.69x the cube handoff's
at a ViT encoder layer of 1370 tokens. It is 0.73-0.82x at an LLM prefill shape from 2048 to
32768 keys. The output's bits are unchanged [HW sweep 2026-10-11, RK3588, 600 MHz]. See
[perf/attention-offload-crossover.md](perf/attention-offload-crossover.md) §"Scores and P as
rows".

### General matmul paths

The general matmul paths still pack A and de-tile C. Their host work has one profile, of a
fully resident 12B F16 model at pp2048. There the A pack and the output read are together at
most 5.3% of prefill wall-clock time [expected, a bound read from one profile]. A device cost
of 0-7% leaves that bound no margin.

## A convolution on the resident matmul

A KxK convolution is one matmul over its unfolded input. The unfold writes `IC·KH·KW` planes of
the `OH·OW` output pixels, each plane one tap's shifted copy with zeros past the edge. The weight
as stored, `[OC][IC·KH·KW]`, is then B. The matmul's channel-planes output is the convolution's
`[OC][OH][OW]`, so neither side needs a transpose. A 1x1 convolution at stride 1 with no pad is its
own unfolded input.

rocket-userspace's `rocket_conv2d_fp16_prepacked` runs the matmul resident, with its weight packed
once, and the unfold on the host across the worker threads. The direct fp16 conv entry scatters its
weight on every call. At pix2pix's eight encoder layers the direct entry summed 148.5 ms against
ONNX Runtime's 47.5 [HW sweep, RK1, 2026-09-28].

The table compares the resident entry with ONNX Runtime 1.27.0's per-node time for the same
layer [HW sweep]:

| pix2pix encoder layer | Resident im2col | CPU, 4 threads | Resident / CPU |
|---|---:|---:|---:|
| c1, 3×256×256 -> 64 | 2.29 ms | 1.98 ms | 1.16 |
| c2, 64×128×128 -> 128 | 5.78 | 11.81 | 0.49 |
| c3, 128×64×64 -> 256 | 4.27 | 10.48 | 0.41 |
| c4, 256×32×32 -> 512 | 3.00 | 14.43 | 0.21 |
| c5, 512×16×16 -> 512 | 2.71 | 6.94 | 0.39 |
| c6, 512×8×8 -> 512 | 1.55 | 2.52 | 0.62 |
| c7, 512×4×4 -> 512 | 1.27 | 2.15 | 0.59 |
| c8, 512×2×2 -> 512 | 1.10 | 2.08 | 0.53 |

**Conditions:** RK1 at 600 MHz, `rocket` 1.3.0, CPU governor `ondemand`, 2026-10-08. The CPU
runs 4 threads on the A76 cores with spinning off. The resident entry uses a three-worker
context.

**Not characterized:** the CPU at thread counts other than 4.

Pinned at `performance`, the same back-to-back calls take 0.58-0.97x the `ondemand` times
[HW sweep].

The first layer loses because three input channels make a poor matmul. Each output pixel moves
112 halves for 3072 MACs (K 48, N 64) [host-computed]. On Depth Anything v2-small's DPT head at
518×518 the ratio runs 0.12-1.43. Its 3x3 convolutions at 19×19 and 37×37 lose, and its stride-2
3x3 at 37×37 wins 8.2x [HW sweep, same operating point].

### Splitting the matmul across the cores

The resident matmul fans one call across its workers in one of two ways. Under the N split each
worker takes a column slice and reads the whole input. Under the M split each worker takes a row
slice and holds the whole weight. The N split therefore re-reads the input once per worker, and
the M split re-reads the weight.

Per call, the M split against the N split [HW sweep, same operating point]:

| Entry and shapes | M split / N split |
|---|---:|
| Forward conv, M 256-1372, N 64-512 (pix2pix c3 and c4, Depth Anything's 19×19 and 37×37 neck) | 0.42-0.65 |
| Forward conv, M 196, N 768 (a ViT patch embedding) | 0.88-0.94 |
| Forward conv, M <= 64, N 512 (pix2pix c5-c8) | 1.3-1.9 |
| ConvTranspose, M >= 1024 (pix2pix up6-up8, SAM's upscalers) | 0.51-0.92 |
| ConvTranspose, M <= 256, B 16x A or more (pix2pix up1-up5) | 1.2-2.5 |

Both resident entries take the M split once every worker gets 64 rows and N is at most 4M. The
rule reproduces every row of the table.

### Inside a session

The times above are for back-to-back calls. Inside ONNX Runtime the same entries run 1.01-1.04x
those times on Depth Anything's neck and 0.95-1.20x on pix2pix's encoder. The CPU governor was
at `performance` for both [HW sweep, RK1, 2026-10-09 UTC]. On Depth Anything they ran
1.27-1.29x while each call's workers started on one core, the one its encoder pins the calling
thread to. The governor note covers that case, and what the governor and the idle state add:
[perf/cpu-governor-and-offload.md](perf/cpu-governor-and-offload.md).

### Accuracy

Integer data whose partials stay below 2048 comes back bit-exact against the direct reference
[HW sweep, RK1, 2026-10-08]. With real data and a bias, the worst error is 2^-14.8 to 2^-11.0 of
each output's magnitude sum. The test threshold is 2^-9 [HW sweep, RK1, 2026-10-08].

### Limits

The resident entry refuses depthwise, the RK3576, and an unfold buffer past 64 MiB. The 64 MiB
bound refuses Depth Anything's two largest head convs, 94 ms of the CPU's 272 ms of
convolutions there.

## Software support

`librocketnpu`'s one-shot entry points pad M==1 up to a height-4 tile (3 zero rows, which
contribute 0 and cannot saturate), compute, and return row 0. The pure planners and the
resident and streaming paths instead require `M%4==0` and reject M==1, because they cannot
cheaply pad pre-packed weights. On those paths the caller must pad a single-vector matmul to
4 rows.

The `ggml-rocket` backend offloads a matmul only at M >= `ROCKET_MIN_M` (default 32, never
below 4).

Two other guards in `librocketnpu` confine the field wrap of Erratum E8 to the thin-channel
case. The `FEATURE_GRAINS` check refuses a tall plane before it wraps. The bank check refuses
an fp16 plane before it reaches a width of 2048.

A short tile also costs more CBUF than its rows suggest. A job pads a sub-input below four
rows up to four (see §"Minimum feature height"). A budget taken at the unpadded rows therefore
admits a tile that the bank check then refuses.

## Evidence

### Minimum feature height

The cosine-metric test is `tests/matmul_correctness_matrix_rocket.c`.

### Per-tile geometry limits

At 32 input channels a 2×2300 plane comes back incorrect on 147199 of 147200 elements. The
test is `tests/conv_width_gate`.

### Row-major activation and output

The exactness test covers 12 shapes with M, K and N all different. The operands are integers,
every element is scored, and the output is prefilled with NaN. The test is
`tests/rowmajor_matmul_probe`.

### Tiles, int8 and fp16 out

The form is exact on 17 test points, with every element scored over a sentinel-filled output.
The test is `tests/rowmajor_tile_probe`.

### Device cost

Each value in the pitch table is taken at 40 rotated reps a process. Two processes ran at 1 and
2 KiB, and four at the rest.

### Cost in a tiled fp16 job

The medians are over 100 rotated reps. The per-rep ratios are bimodal by position in the
rotation. Each figure is therefore the median of each configuration's own times, and each
configuration ran at every position equally often. The test is
`tests/rowmajor_kacc_chain_probe`.

### A convolution on the resident matmul

The table's resident entry ran two passes of nine warm calls each, in `tests/conv_model_bench.c`.
The `performance` comparison covers 22 shapes.

The accuracy sweep covered 18 shapes along stride, pad, dilation, an explicit extent, 1x1, a
patch embedding, 1xK, IC 3 and unaligned OC and `OH·OW`. Every shape ran with one and three
unfold threads. pix2pix's and Depth Anything's layers also ran under both splits and
`ROCKET_KACC=0`. The test is `tests/conv_resident.c`.

### Excluded causes

- **The `surf_stride < 0` clamp, for the minimum feature height.** The clamp is present and
  correct, yet the matmul still produces incorrect output at height 1.
- **DRAM bank aliasing, for the row-major write cost.** A pitch 16 bytes off a power of two
  (2052, 4100, 4160 and 8196 elements) costs what its power-of-two neighbor does.
