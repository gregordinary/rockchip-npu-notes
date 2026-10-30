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
([encodings/tile-layouts.md](encodings/tile-layouts.md)). Software must keep K and N aligned as
the table gives. M needs no alignment, given the feature surface stride of §"Feature surface
stride".

| datatype | K must divide | N must divide |
|---|---:|---:|
| fp16 | 32 | 16 |
| int8 | 32 | 32 |
| int4 | 32 | 64 |

bf16 follows fp16 at K%32, N%16. tf32 is K%16, N%16: a 4-byte element halves the K-group.
int16-exact follows int8 at K%32, N%32.

N-alignment grows as the weight N-group grows (fp16 16 -> int8 32 -> int4 64).

> [!CAUTION]
> A wrong N-alignment produces incorrect output with no error [HW sweep]. For example, int8
> with `N=16` over-reserves a 32-kernel group, and the NPU, reading 16 kernels, disagrees.

### Feature surface stride

The CNA reads a feature cube in lines of four rows. `LINE_STRIDE` in `CNA_DMA_CON1` (`0x107C`)
is the distance between lines, `width × 4` atoms in a packed cube. The CNA reaches the next
channel surface at the line stride plus `SURF_STRIDE` in `CNA_DMA_CON2` (`0x1080`) [HW sweep
2026-10-21]. Software must program the surface stride of a cube of H packed rows as
`width × (H − 4)` atoms. Below four rows the value is negative, and software must write it as a
28-bit two's complement.

A matmul's rows are the cube's height at width 1, so its surface stride is `M − 4`. A matmul
can instead lay M along the width at height 1, with a surface stride of `−3M`.

**Conditions:** RK3588, kernel 7.2.9, `rocket` 1.3.0, NPU clock 928 MHz, one task. fp16 at K 128
to 2048 and N 64 to 256. int8 at K 128 and 1024, N 64 and 256.

With that stride, an fp16 or int8 matmul task is bit-exact at every M measured: 1-17, 31, 33, 63
and 64. fp16 is also bit-exact with M along the width [HW sweep 2026-10-21]. int4, int16, bf16
and tf32 share the generators' cube geometry and are not characterized below `M%4==0`.

A convolution reads its input through the same stride, with H the input height. With
`width × (H − 4)`, a one-task fp16, fp16 depthwise or int8 convolution is bit-exact at input
heights 1, 2 and 3 [HW sweep 2026-10-22]. No four-row minimum applies to the input. The int8
depthwise convolution is not characterized below four input rows.

**Conditions:** RK3588, kernel 7.2.9, `rocket` 1.3.0, NPU clock 928 MHz, one task. fp16 at input
channels 8, 32 and 96, input widths 1, 7 and 64, kernels 1×1 and 3×3, strides 1 and 2. int8 at
input channels 16, 32 and 96, the same widths and kernels, stride 1. fp16 depthwise at 64
channels, 3×3.

> [!CAUTION]
> Erratum E3: the surface stride `line_stride × (H/4 − 1)`, computed in integer arithmetic, is
> correct only when H is a multiple of 4. At any other height the division truncates, and a
> negative result clamped to 0 is also incorrect. A matmul programmed this way produces incorrect
> output at every M that is not a multiple of 4. A convolution with its stride clamped at 0
> produces incorrect output below four input rows whenever its input spans more than one
> channel group. It is bit-exact at M 4, 8, 12, 16 and 64 [HW
> sweep 2026-10-21]. Software must compute the stride as `width × (H − 4)`.

Mesa computes `line_stride × (H/4 − 1)` in floating point, which equals `width × (H − 4)` at every
height of four or more [source-confirmed: Mesa `rkt_task.c`]. An external userspace programs
`−3M` with M along the width [source-confirmed: `rknpu2_matmul_open`, [SOURCES.md](SOURCES.md)].

A task that reads A as plain rows (§"Layouts") programs no surface stride, and is bit-exact at
the same M [HW sweep 2026-10-21].

Padding M from 1 to 4 adds MAC work and no weight traffic [host-computed].

## Tiling

A task's input tile must fit the CBUF data banks. Its weight tile either fits the weight banks
whole, the whole-tile form, or streams through them (§"CBUF weight streaming"). Large
matmuls are therefore tiled three ways:

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

### CBUF weight streaming

In [CBUF weight streaming](GLOSSARY.md), the CNA streams a weight cube that is larger than the
weight banks a task programs. The input tile stays in the data banks while the 32-kernel weight
groups cycle through the weight banks. One task therefore covers a whole N slice. fp16 and
int8 tasks stream bit-exactly with weight cubes of 2-20 MiB [HW sweep 2026-10-21]. The
streamed form needs far fewer tasks than the whole-tile form. `512×3840×4096` in fp16 takes 12 streamed
tasks against 512 whole-tile tasks [host-computed].

**Conditions:** RK3588, kernel 7.2.9, `rocket` 1.3.0, NPU clock 928 MHz. fp16 at K 384-15360,
M tiles of 32-256 rows and N 208-4096. int8 at K 640-16384, M tiles of 4-256 rows and N to 8160.

Software must program the weight banks to hold at least one whole 32-kernel group,
`32 × K × element bytes`, and at least two banks. One bank works only where the group fills
it exactly, as fp16 at K 512 does. An int8 task at that minimum can hang (Erratum E11), and
software must give it one bank more. Above the minimum, the weight-bank count changes the task
time by under 3% (2 to 8 banks at K 384-2560) [HW sweep 2026-10-21].

> [!CAUTION]
> Erratum E11: a streamed task below the weight-bank minimum never completes. The job runs
> until the driver's watchdog retires it (~500 ms on the RK3588), the fence signals as if the
> job had completed, and the output is not written. The next job produces correct output.
> fp16 with one weight bank hangs this way at K 384, 896 and 1152. int8 hangs with two weight
> banks at K 1120 and 1376, whose atoms are 1.09 and 1.34 banks, and not with three [HW sweep
> 2026-10-21]. Software must program at least two weight banks, enough banks for one
> whole group, and for int8 one bank more.

A streamed task has three further requirements:

- Software must program `FEATURE_GRAINS` (`CNA_CONV_CON2`) to the tile's height plus one. At
  K 512 and M 128, a value of 17 gives ~1000 incorrect elements of 262144 [HW sweep 2026-10-21].
- Software must program `FC_DATA_BANK` to 0. Values 2-5 hang a streamed task until the
  watchdog ([encodings/cbuf-reuse.md](encodings/cbuf-reuse.md)).
- Software must clear `WEIGHT_REUSE` on a streamed task
  ([encodings/cbuf-reuse.md](encodings/cbuf-reuse.md)).

The task count it removes and the per-core weight bandwidth are in
[perf/not-mac-bound.md](perf/not-mac-bound.md) §"Streamed weights".

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
hardware-tested correct to K = 10240 [HW sweep]. In the whole-tile form, K tiles to a few
hundred only because the output tile (Mt×Nt) fixes how much CBUF is left for Kt. Shrinking Mt
and Nt grows Kt and collapses nKt. int4 on Gemma's `K=3840` reaches `nKt=1` (single-pass, zero
readback K-accumulation) at Mt=Nt=64.

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

**Conditions:** RK3588, `rocket` 1.3.0, NPU clock ~816 MHz.

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

**Conditions:** RK3588, `rocket` 1.3.0, NPU clock ~816 MHz.

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

**Conditions:** RK3588, `rocket` 1.3.0, NPU clock ~816 MHz, 2026-09-26. Each wait is on a
4 KiB BO that the job lists as an output, and every configuration's BOs are the same size.

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

**Conditions:** RK3588, `rocket` 1.3.0, NPU clock ~816 MHz, 2026-09-26.

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
32768 keys. The output's bits are unchanged [HW sweep 2026-10-11, RK3588, ~816 MHz]. See
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

**Conditions:** RK1, NPU clock ~816 MHz, `rocket` 1.3.0, CPU governor `ondemand`, 2026-10-08.
The CPU runs 4 threads on the A76 cores with spinning off. The resident entry uses a
three-worker context.

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

`librocketnpu`'s matmul generators program the feature surface stride as `width × (H − 4)`, so
a task computes at any M. Its planners still require `M%4==0`. Its one-shot entry points pad
M==1 up to 4 rows (3 zero rows, which contribute 0 and cannot saturate), compute, and return
row 0. The resident and streaming paths reject M==1, and on those paths the caller pads a
single-vector matmul to 4 rows.

The `ggml-rocket` backend offloads a matmul only at M >= `ROCKET_MIN_M` (default 32, never
below 4).

Two other guards in `librocketnpu` confine the field wrap of Erratum E8 to the thin-channel
case. The `FEATURE_GRAINS` check refuses a tall plane before it wraps. The bank check refuses
an fp16 plane before it reaches a width of 2048.

`librocketnpu`'s fp16 and int8 generators stream a weight cube larger than its banks. They
refuse a task below the weight-bank minimum, which `npu_stream_weight_banks()` computes (the
int8 generator one bank above it), and refuse `WEIGHT_REUSE` on a streamed cube. The fp16
planner chooses a streamed plan by default (`ROCKET_MM_STREAM`, `=0` selects the whole-tile
plan). The int8 planner streams under `ROCKET_MM_STREAM_INT8=1`.

A short convolution tile also costs more CBUF than its rows suggest. The convolution path
writes the surface stride as `width × (H − 4)` clamped at 0, and pads a sub-input below four
rows up to four. A budget taken at the unpadded rows therefore admits a tile that the bank
check then refuses.

## Evidence

### Feature surface stride

The test is `tests/msmall_probe`. It compares exact fp32 output from integer operands in
[-3, 3] against a NaN-stamped output, and runs each M twice with different operands. The
strides were varied one at a time, M 1-8 at K 128 and N 64:

| Geometry | Line stride | Surface stride | Result |
|---|---|---|---|
| Width 1, height M | `4` | `line × (H/4 − 1)`, clamped at 0 | Incorrect at every M%4 != 0 |
| Width 1, height M | `4` | `line × (H/4 − 1)`, unclamped | Incorrect at every M%4 != 0 |
| Width 1, height M | `4` | `−3M` | Bit-exact at M 1 only, where `−3M = M − 4` |
| Width 1, height M | `4` | `M − 4` | Bit-exact at every M |
| Width M, height 1 | `4M` | 0 | Incorrect at every M |
| Width M, height 1 | `4M`, `0x40` or `0x200` | `−3M` | Bit-exact at every M |
| Width M, height 1 | `0x200` | `−2M` or `−4M` | Incorrect at every M |

`FEATURE_GRAINS` at 2 or at M+1 makes no difference. The line stride makes none at height 1,
where the cube has a single line. The `M − 4` row was then run at M 1-17, 31, 33, 63 and 64, at
K 128, 1024 and 2048. The `−3M` row ran at those M at K 128, and at M 1-8, 12 and 16 at K 1024.
The row-major form ran at M 1-8, 12 and 16 at K 128, 1024 and 4096, and at the full M list at
K 2048.

The int8 configuration is `MSP_DT=int8`, with C2=16 input and int32 output.

The formula leaves every program at a height that is a multiple of 4 unchanged. The six matmul
generators' programs at 5555 such shapes are byte-identical under both formulas
[host-computed].

The convolution tests are `tests/conv2d_fp16_rocket` and `tests/conv2d_int8_rocket` with one
shape on the command line. Inputs are small hashed or random integers, and the output is
compared exactly against an integer reference. A debug variable, `ROCKET_CONV_NOPAD=1`, runs the
task at the real input height instead of padding it to four rows with zeros. Each shape ran three
ways: padded, unpadded with the signed stride, and unpadded with the stride clamped at 0:

| Configuration | fp16 and fp16 depthwise | int8 |
|---|---|---|
| Padded to four rows | 111 of 111 bit-exact | 54 of 54 bit-exact |
| Unpadded, `width × (H − 4)` | 111 of 111 bit-exact | 54 of 54 bit-exact |
| Unpadded, clamped at 0 | Bit-exact at 8 input channels (one group), incorrect at every shape with more | Incorrect at 32 and 96 input channels; bit-exact at 16, which the entry pads to 32 with zero weights |

The int8 rows ran through the library entry, which pads the input channels to 32, and 36 of
them also through the generator directly. The clamped configuration ran only at input widths 1
and 7, because at width 64 it reads past the input buffer. At input heights 4, 5 and 8 all three
configurations are the same program, and all are bit-exact. The convolution generators'
programs at 5400 shapes of input height 4 or more are byte-identical under both formulas
[host-computed]. Another RK3588 project that reports M 1-3 incorrect writes the truncating
formula ([SOURCES.md](SOURCES.md)).

The cosine-metric test that first recorded the failure at M 1 is
`tests/matmul_correctness_matrix_rocket.c`.

### CBUF weight streaming

The tests are `tests/weight_stream_probe` and `tests/gemv_bw_probe`. Their operands are
integers, so every sum is exact, and each element is compared for equality with a double
reference over a NaN-stamped output. The fp16 shapes cover K 384-2560, M tiles of 32-256 rows,
N 1024-4096 and weight cubes of 2-20 MiB. The int8 shapes cover K 3072 and 8192 and N to 8160.
The weight-bank sweep covers 1 to 8 banks at K 384-2560. A 64×256×64 canary job runs and is
scored after every streamed job, and after each hung task it produced correct output.

The library's streamed plans were then scored through its resident entries, every element
exact. fp16 ran `tests/matmul_prepacked_crossm_rocket` at six model shapes, K 3072-15360 and
N 1024-12288 at M 256, 512 and 768, with integer operands. int8 ran
`tests/matmul_int8_prepacked_rocket` at 18 shapes, K 640-16384, M 4-768 and N 512-9216, against
the one-shot program. The 14 of them within 2^31 MACs were also scored against a host int64
reference.

The int8 weight-bank map ran single-K-pass tasks at N 1024, with M and K chosen to set the
weight banks the task is left:

| K | Atom, banks | M | Weight banks | Result |
|---:|---:|---:|---:|---|
| 1120 | 1.09 | 256 | 2 | Every task timed out |
| 1120 | 1.09 | 224 | 3 | Bit-exact |
| 1376 | 1.34 | 200 | 2 | Every task timed out |
| 1376 | 1.34 | 180 | 3 | Bit-exact |
| 2048 | 2.00 | 132 | 2 | Bit-exact |
| 2080 | 2.03 | 120 | 3 | Bit-exact |
| 3072 | 3.00 | 84 | 3 | Bit-exact |

Rows at a spare bank or more (K 1024, 2048, 2080 and 3072) were bit-exact too. The map does not
separate the alignment of the atom from K%64, because both failing K are odd multiples of 32.

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

- **DRAM bank aliasing, for the row-major write cost.** A pitch 16 bytes off a power of two
  (2052, 4100, 4160 and 8196 elements) costs what its power-of-two neighbor does.
