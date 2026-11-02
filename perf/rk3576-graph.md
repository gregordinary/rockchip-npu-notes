# Whole-network inference on the RK3576

MobileNetV1-224 at int8 runs end to end on the RK3576 and returns TFLite's own label. The graph
is 27 convolutions, a PPU pooling layer and a classifier. Every one of the 29 compute layers is
bit-exact against a CPU model of the RK3576's arithmetic. The whole graph's wall-clock time per
inference depends on which optimizations are in place:

**Conditions:** H96 MAX M9, MobileNetV1-224 int8. Each row is its own measurement, taken as its
optimization was added, from 2026-07-29 (the bottom row) to 2026-08-03 (the top row).

| Configuration | Wall-clock time per inference |
|---|---:|
| The pooling layer inside the kick, so the whole graph is one kick in one submit | **4.9-5.0 ms** |
| Resident weights, tensors kept in cube layout between layers, each chained run of cube-joined layers as one hardware kick, and the pooling layer resident, reading its input as a cube and retiring on the PPU's own completion | 5.1 ms |
| The same with the pooling layer transient, scattering its input on the host, allocating its buffers per call and retiring at the `dpu_grace_us` deadline | 6.2 ms |
| One submit per layer | 10.4-10.5 ms |
| The transposes at every layer boundary | 21 ms |
| No resident weights at all | 98.8 ms |

The sections below take that difference apart and measure each part of it. The method, the
controls and the run detail behind each section are in
[../chips/rk3576-validation.md](../chips/rk3576-validation.md), under the same section titles. The
datapaths and the hazards the graph runs on are in [../chips/rk3576.md](../chips/rk3576.md) and
[../chips/rk3576-hazards.md](../chips/rk3576-hazards.md).

## A whole network

The graph planner places and joins a network's layers for every frontend. The TFLite
delegate is one frontend, and runs the plan on unmodified `.tflite` files.

### The graph planner

The placement and join passes are in `graph/rocket_graph_rk3576.c`:

- One buffer per add and per concatenation, with a slice per operand
- The declared tails
- The shared surfaces
- The adjacent-pair join, with its refusal buckets
- The caller of the chained-run finder

The passes run over a frontend-neutral layer description:

- Kind
- Geometry
- The operand graph, by producer
- The resident handles
- One flag for the thing no rule there can derive: whether the caller prepares that layer's
  input on the host

The planner's caller holds the rest:

- The weights
- The tensors
- The ping-pong
- The skip lifetimes
- The de-tile points

The planner is a separate target that the frontends link. It is not part of `librocketnpu`,
which holds no graph-execution object. The chained-run finder (`rocket_chain_plan_rk3576()`) is in
the library because it is pure, and it is the one part a frontend cannot hold itself.

Twelve refusal classes are closed, and every one is a host-buffer rule rather than a hardware
rule. The frontends share the planner rather than copy it, because a second copy would diverge
where being wrong is silent. A frontend that uses the planner gets every row of the
MobileNetV1-224 table above. A frontend that calls only the op entries gets about a twentieth of
the demonstrated throughput. [HW sweep, H96 MAX M9, 2026-08-04]

### The TFLite delegate

A second caller runs the planner: the TFLite delegate, `tflite-rocket`, on real `.tflite` files.
It lowers a delegated partition onto the same neutral description and runs the plan. An
unmodified `.tflite` therefore reaches the RK3576 through the ordinary external-delegate API,
rather than only through a hand-built blob.

**Conditions:** H96 MAX M9, kernel 7.1.6, NPU 786 MHz, warm, on a real image, 2026-08-04. The
delegate column includes the interpreter's own invoke and the transposes at the partition
boundaries. The CPU column is the same interpreter with no delegate. The last three columns are
the nodes claimed, the adjacent-pair joins and the hardware kicks that the plan produced.

| model | delegate | CPU | claimed | joins | kicks |
|---|---:|---:|---:|---:|---:|
| MobileNetV1-224 | 5.2 ms | 71.7 | 29/31 | 28/28 | 1 |
| MobileNetV2-224 | 6.8 ms | 45.3 | 64/66 | 63/63 | 1 |
| ResNet-18-224 | 8.8 ms | 167.1 | 36/38 | 27/27 | 1 |
| Inception V1-224 | 7.4 ms | 180.5 | 81/83 | 53/53 | 1 |
| Inception V3-299 | 34.5 ms | 669.3 | 130/132 | 90/94 | 16 |
| SSD-MobileNetV2-COCO | 17.0 ms | 109.0 | 87/111 | 70/82 | 6 |
| EfficientDet-Lite0-320 | 55.5 ms | 123.1 | 238/267 | not re-read | not re-read |

The EfficientDet-Lite0 row is from 2026-09-27, on `rocket` 1.6.0 and kernel 7.2.3 [HW sweep,
H96 MAX M9]. Both CPU clusters were pinned to `performance`, and the per-axis gain was in the BS
shift word (§"Per-axis gain on the BS shift word"). No layer is refused, and the 29 unclaimed
nodes are op kinds the delegate does not claim. Its joins and kicks were not re-read on that
build.

Every classifier returns the interpreter's own top-1. The delegate attempts the packed-image stem
only at a zero input zero point, where the output stays a cube.

### Claim-time refusals

A `DEPTHWISE_CONV_2D` of four or fewer channels has no form on the RK3576. The packed-image first
conv folds the kernel's columns into the channel axis, so nothing is left to be depthwise over.
The library refuses `direct_datapath` on a depthwise descriptor rather than ignoring it.
[HW sweep, H96 MAX M9, 2026-08-04]

The detectors contain two node kinds that no hand-built graph produces: that depthwise
convolution, and a `QUANTIZE` on a narrow tensor. The `QUANTIZE` meets the same limit through its
lowering, since the requantization edge's natural form is a depthwise 1×1 identity. On four or
fewer channels it takes the direct identity instead, which is the same arithmetic and sixteen
weights. An SSD box-encoding branch has exactly both shapes. [HW sweep, H96 MAX M9, 2026-08-04]

The delegate must answer both at claim time. A layer the library will not pack is a Prepare
failure, and a Prepare failure fails the whole model. The framework runs an unclaimed node
itself. [HW sweep, H96 MAX M9, 2026-08-04]

A claim-time refusal is cheaper than a late one by more than a prepare. Before it claims a node,
the delegate checks every bound that the library enforces and that the shape decides:

- The pooling plan
- The per-axis gain ramp
- Through `rocket_conv2d_int8_plan_rk3576()`, the convolution geometry, the resident weight
  slice and the row window

> [!NOTE]
> The pack never reaches the row window, and only the run's per-task plan does. A model past
> the row window builds cleanly, passes `AllocateTensors`, and throws at every `Invoke`. That
> failure comes later than a Prepare failure, and past any load-time validation a caller can do.

### Detector accuracy

**Conditions:** mAP@[.5:.95] over 500 COCO val2017 images, from the same interpreter with and
without the delegate, H96 MAX M9, 2026-08-04. EfficientDet-Lite0's per-axis layers run on the
shift-0 ramp, the `C` ramp with the BS shift word at zero (`ROCKET_RK3576_PC_SHIFT=0`), with
`ROCKET_RK3576_PC_INERT` at its default. With the gain in the BS shift word, the default per-axis
path, it scores 0.2818 (§"Per-axis gain on the BS shift word").

Both detectors score at CPU parity:

| model | delegate | CPU | delta |
|---|---:|---:|---:|
| SSD-MobileNetV2 | 0.2617 | 0.2631 | -0.0014 |
| EfficientDet-Lite0 | 0.2826 | 0.2823 | +0.0003 |

> [!NOTE]
> A detector's output is an NMS-ordered list, so one count of arithmetic drift reorders or drops
> a box. An element-wise diff against the CPU therefore says nothing about a detector's accuracy,
> and mAP is the comparison a detector admits.

## Per-axis requant on a graph

A per-axis model gives each output channel its own weight scale. The sections below give how a
graph applies those scales and where its requant departs from TFLite's. They also give three
rules a whole network depends on that per-op shapes do not reach.

### Per-channel C minimum

**Conditions:** H96 MAX M9, per-axis ResNet-18 and EfficientDet-Lite0 through the TFLite
delegate, on the shift-0 ramp, 2026-08-04. On the default per-axis path, the BS shift word sets
each task's base gain from its largest live scale. Only a live channel below 1/65534 of that
scale clamps there (§"Per-axis gain on the BS shift word").

The per-channel `C` has a minimum of one, and below it a channel computes at the tile's gain
instead of its own. A tile is one task, and a task has one `OUT_CVT` shift. A channel therefore
reaches its own scale through the int16 `C` in its coefficient group and nothing else. The base
gain is already the smallest the tile admits, and every channel's `C` must fit that channel's
own accumulator bound. A channel whose scale sits below half the base therefore needs a `C`
under one, and the field's minimum is one.

> [!CAUTION]
> A channel whose `C` clamps at one is wrong by a factor, and its relative gain error exceeds
> 100%. The output is the correct size with plausible values, and no error is reported.
> Software must find every clamped channel before it submits a per-axis layer.

The packer's log measures 311% and 8150% on two per-axis ResNet-18 layers, and past 100% on 7 of
EfficientDet's 182 per-channel layers. The clamp is the ramp's own boundary rather than a chosen
threshold. Above it, `C` is an integer and the resolution is `0.5/C`, which the tile search
improves at the cost of more submits. Below it there is no `C` to choose.

A clamped channel is exactly one whose relative gain error exceeds 100%, and the library refuses
its layer unless the channel's output cannot vary (§"Inert clamped channels"). The pure call
`rocket_conv2d_int8_perchannel_plan_rk3576()` answers that question. The delegate asks it at
claim time, where a refusal costs one node that the framework runs itself. The median
EfficientDet layer is 0.021%.

The depthwise path is the one affected. A depthwise layer is one task at any channel count, so
its tile cannot be narrowed. Of the 7 clamped layers, 6 are depthwise (1152, 1152, 1152, 672,
240, 32) and 1 is direct: the stem, already at the emitter's minimum of 32. With
`ROCKET_RK3576_PC_ALLOW_CLAMP=1` the layer runs anyway, at a cost of 0.0428 mAP on EfficientDet:
0.2395 against the CPU's 0.2823, where refusing scores 0.2809.

### Inert clamped channels

**Conditions:** H96 MAX M9, EfficientDet-Lite0 through the TFLite delegate, 500 COCO val2017
images per configuration, the shift-0 ramp, 2026-08-04. The default per-axis path puts the gain
in the BS shift word and programs every channel whose output cannot vary as its byte. Its time
and mAP are in §"Per-axis gain on the BS shift word".

A clamped channel in these models is a pruned one. Each of the 111 clamped channels in the six
depthwise layers, and each of the stem's four, has an all-zero filter. An all-zero tensor is
quantized against the quantizer's minimum scale (TFLite's 1e-6). The channel's scale therefore
sits four to five orders of magnitude below its neighbors', while its int32 bias still has the
layer's magnitude. A scale that far below its neighbors' puts the channel below half the tile's
base gain.

A clamp is a defect only where it changes the output byte. A channel with no live weight
produces one accumulator value, the zero-point fold `A`, at every output position. Its MAC term
is zero for any feature cube. Whether such a channel's output byte is correct is therefore
decidable exactly, with no bound and no sampling.

For 105 of those 111, the fold is large and negative, and saturates to -128 under either gain.
The library's rule tests saturation rather than equality: both gains must drive the fold past
the same saturation limit by more than one count. The answer then does not depend on the DPU's
rounding, which the rule models on the host rather than running it. A clamped channel with a
live weight is always refused. Its accumulator ranges over `+-128*sum|w|` around `A`, and a gain
wrong by a factor is wrong somewhere in that range.

The rule takes EfficientDet-Lite0 from 7 refusals to 4. It accepts the stem, the first
depthwise layer and the 240-channel layer. On their pruned channels, all three are exact
against an exact per-axis requant [HW sweep, H96 MAX M9, 2026-08-04]. Their non-clamped
channels are all within 0.077-0.097% of their own gain.

Accepting the clamped channels raises the detector's score slightly, and their per-axis
arithmetic costs no accuracy. With them refused, EfficientDet-Lite0 scores 0.2809 against the
CPU's 0.2823. With them accepted on the depthwise path only, it scores 0.2832, and with the
default full acceptance, 0.2826 [HW sweep, H96 MAX M9, 2026-08-04].

`ROCKET_RK3576_PC_INERT` selects the acceptance, and is read only on the shift-0 ramp. Its default
is full acceptance, `0` refuses every clamped channel, and `dw` restricts the acceptance to the
depthwise path.

A refused layer splits the delegated partition it sits in, so its cost depends on the graph
rather than on the node. With the clamped channels refused, EfficientDet-Lite0's head is a
12-layer and a 28-layer partition, with 5394 KiB of cube joins between them. With them
accepted, the head is one 43-layer partition with 7594 KiB. The claimed-node counts barely
change (231, 233, 234 of 267), and inference time changes by 17%:

| Setting | claimed | head partition | inference time | vs CPU TFLite |
|---|---:|---|---:|---:|
| `PC_INERT=0` | 231 | 12 + 28 layers, 5394 KiB joined | 99.3 ms | 1.23x |
| `PC_INERT=dw` | 233 | 42 layers, 6794 KiB | 90.3 ms | 1.35x |
| **`PC_INERT` default (full)** | **234** | **43 layers, 7594 KiB** | **82.8 ms** | **1.47x** |

**Conditions:** CPU governor pinned to `performance` [HW sweep, H96 MAX M9, 2026-08-04].

Inference time separates the two acceptance modes, and the mAP does not. Their 7.5 ms
difference is twelve times the run-to-run spread. Their 0.0006 mAP difference is inside what
one count of drift does to an NMS-ordered detection list [HW sweep, H96 MAX M9, 2026-08-04].

### Channel-range splits

On the shift-0 ramp, splitting a depthwise layer into channel ranges removes the clamp from 3 of
the 6 clamped depthwise layers [host-computed, 2026-08-04]. A depthwise convolution is one task
because output channel `c` is bound to input channel `c`. A split gives each channel range its own
task and its own `OUT_CVT` shift. Each range is a multiple of the output cube's 16-channel atom.

The split clears op#15 at a uniform tile of 96, and ops #44 and #52 at 32 and 16. A
non-uniform partition clears them at 3, 6 and 3 programs, built greedy-longest from the left,
which a planner can emit. The other three each contain an irreducible 16-channel atom: one that
holds both the channel that sets the base gain and the smallest-scale channel. No 16-aligned
partition clears them at all. One of them (op#48) gets worse as the tile shrinks, with 1
clamped range at 1152 against 4 at 32 [host-computed, 2026-08-04].

A depthwise layer cannot reorder its channels. A consecutive range therefore narrows the scale
spread only where the model's channel order happens to be locally scale-ordered. Whether a
split helps depends on the model, not on the tiling.

### Channel order of a per-axis output

The direct per-axis path of `librocketnpu` sorts a layer's output channels by scale. Each
output-channel tile then spans as little of the layer's scale range as the tiling allows. A
tile's channels occupy consecutive slots of the output. A sorted handle therefore writes its
output channels in scale order, not in the model's order. The host de-tile puts a
materialized tensor back in model order, so a per-op result that is materialized shows no sign
of the sort.

> [!CAUTION]
> A layer that reads its input as a cube takes the channel order as stored. Fed a scale-sorted
> output, it produces output of the correct size with incorrect values, and reports no error.
> Software must hand the next layer a cube in the model's channel order.

With one output-channel tile, the sort changes nothing: that tile spans every channel in any
order. The base gain and each channel's `C` are the same values in different slots. The
single-tile path therefore does not sort, and writes its output in model order. Multi-tile
handles keep the sort, which is where their accuracy comes from. Their cube is refused in any
case, because each tile has its own output BO. `r76_cube_shape_ok()` also checks the
invariant, where the cube join depends on it.

With both per-axis defects handled, per-axis ResNet-18 agrees with the CPU's top-1 on 89 of 100
images, against the per-tensor build's 93 of 100. EfficientDet is at CPU parity. The
remaining gap is the per-axis requant's own accuracy on near-ties, which is the expected regime
[HW sweep, H96 MAX M9, 2026-08-04].

### Per-axis gain on the BS shift word

**Conditions:** H96 MAX M9, `rocket` 1.6.0, EfficientDet-Lite0 through the TFLite delegate,
2026-09-27.

As on the RK3588, the BS shift word contains a shift field for each sign
([../chips/rk3576-regcmd.md §"DPU shift operand
(`0x5024`)"](../chips/rk3576-regcmd.md#dpu-shift-operand-0x5024)) [HW sweep, H96 MAX M9].

> [!CAUTION]
> A shift programmed in one sign field alone scales the other sign's products wrongly, and the
> output has the correct size and incorrect values. Software must program the shift in both
> sign fields.

`librocketnpu` programs the per-axis gain in the BS shift word. Each task's largest live scale
takes `C` = 32767. The BS shift is the smallest at which `((acc + A)*C) >> s` fits int32 at
every live channel's accumulator bound. A channel whose exact output is one byte over its whole
reachable accumulator is programmed as that byte, through its own `A` and `C`.

On EfficientDet-Lite0 those constant-byte channels number 345 over 28 layers: all-zero filters,
and dead channels driven past the saturation limit. They are every channel that the shift-0
ramp, the `C` ramp with the shift word at zero, clamps. Only a live channel below 1/65534 of its
task's largest live scale is refused. The per-axis path is bit-exact against the extended host
model [HW sweep, H96 MAX M9, `rocket` 1.6.0, 2026-09-27].

Under the shift-0 ramp, the output-channel split costs more than the refusals. The ramp refuses
4 of EfficientDet-Lite0's 182 per-axis layers and cuts 11 direct layers into 133 tiles. A
multi-tile layer cannot produce a cube, so 6 joins are refused too. Of an 83.1 ms inference, the
refusals cost 6.3 ms and the tiles 20.1 ms.

With the gain in the shift word, no layer is refused or split. With both CPU clusters pinned to
`performance`, the model runs in 55.5 ms against 83.6 (0.661x paired), 2.22x the CPU's 123.1 ms
[HW sweep, H96 MAX M9]. Its mAP@[.5:.95] is 0.2818 against the CPU's 0.2823 over 500 val2017
images. Setting `ROCKET_RK3576_PC_SHIFT=0` restores the shift-0 ramp. Its detections are then
byte-identical to those of a build without the shift word.

### Distance from TFLite

**Conditions:** MobileNetV1-224, per-tensor int8, H96 MAX M9.

The DPU's `OUT_CVT` converter and TFLite's requant are fixed-point approximations of one
real-valued scale, with different multiplier widths and tie rules
([../encodings/out-cvt-converter.md](../encodings/out-cvt-converter.md#the-cvt_round-tie-rule)).
They therefore disagree on the elements near a rounding boundary.

Against TFLite the graph differs on 0.095% of elements, every one of them by a single count,
and that difference is the requant's. The pooling layer and the classifier match exactly. When
each layer takes its own previous output rather than TFLite's, the disagreements compound down
the graph. At the classifier 76% of the logits are bit-identical, and the top-1 and the whole
top-5 still agree.

The fused RELU6 costs nothing. Every intermediate tensor is quantized at scale 6/255 with its
zero point at the bottom of the range. The activation's clamp is therefore exactly the int8
saturation the DPU already applies.

### CBUF allowances, SAME padding and the stem

A whole network depends on three rules that per-op shapes do not reach. The first is that the CNA's
trailing pad is the pad its last window consumes, derived from the output extent. TFLite's
asymmetric SAME padding is therefore programmed as an output extent, with no border copied into the
cube chain ([../chips/rk3576.md](../chips/rk3576.md) §"Asymmetric padding as an output extent").

> [!CAUTION]
> On the depthwise path, the two low CBUF allowance settings are not honored. A task whose
> window selects one writes incorrect output past the budget it receives, and reports no
> error. Software must not select either setting on the depthwise path.

On the direct path, whether those settings are honored depends on the resident weight footprint.
On the depthwise path no footprint threshold fits at all, and the library declines both
settings. See [../chips/rk3576.md](../chips/rk3576.md) §"The symmetry audit on the CNA" and
§"CBUF allowances on the depthwise path".

The int8 first convolution on a packed image has bounds that together encode the ONNX
symmetric-SAME convention. A TFLite SAME stem has `pad_left = 0`, so it does not meet them
directly. MobileNetV1-224 therefore runs its stem on the direct path, at the model's own three
channels. The direct path takes three channels because the int8 direct cube is a 32-channel MAC
group at every channel count ([../chips/rk3576.md](../chips/rk3576.md) §"Narrow input-channel
counts on the direct datapath").

Widening the stem to eight channels is also exact. The added channels hold the input zero point
against zero weights. Such channels add nothing to either half of the coefficient `A` fold, for
any zero point.

The stem has a second lowering. Shifting the input by one pixel against a pad of one moves the
sample grid by a whole stride. The shifted grid reproduces TFLite's bit-exactly in the interior.
That lowering is implemented and tested, and it is slower than a resident widened stem
([../chips/rk3576.md](../chips/rk3576.md) §"TFLite stem through an input shift").

## Host terms of an inference

The host terms of an inference are the weight packing, the output stamp and the transpose
between row-major tensors and the cube. The sections below measure each, and the change that
reduces it.

### Inference time with per-call weight packing

**Conditions:** MobileNetV1-224, H96 MAX M9, weights packed on every call, a transpose at every
layer boundary, 40 submits of one task each, 2026-07-29.

Every layer of the graph runs on paths with no wide-output hazard. The graph is int8
throughout, so it needs neither the guard against a corrupted next submit nor a power cycle. The
atom drop that the int32 writers show did not appear. The measured inference time therefore
contains no redo and no power cycle [HW sweep, H96 MAX M9].

Under these conditions the inference time is host arithmetic over the weights, not dispatch.
The three layers that cost most, the stem and the two 1024-channel 1×1 convolutions, take 5, 1
and 1 of the 40 submits. Output-channel tiling is not what those layers spend time on: each is
one output-channel tile, and the stem's five submits are its row tasks ("Join pass", below).

`ROCKET_RK3576_INT8_PROF=1` splits a call into seven buckets:

- Sums
- Feature pack
- Weight pack
- Coefficients
- Plan+gen
- Submit
- De-tile

Across the graph the submit is 1-16% of a layer and 3% of the two most expensive, against a
host share of 84-99%. These jobs take 0.13-0.25 ms each. The ~340 us per-submit minimum applies
to a round trip that waits for a whole output to drain, and they are well under it.

> [!NOTE]
> A layer's submit count does not give its submit time. Measure the submit time before
> attributing a layer's cost to dispatch.

The sections below remove these terms. Resident weights remove the weight arithmetic, the cube
layout removes the transposes, and the kick removes most of the submits. With all three in
place, the host share is 6-12%, and the two remaining terms are the submit and the write guard
(see "Composition of a graph's wall-clock time").

The host work is two loops over the weight tensor, and neither depends on the input. The
per-output-channel filter sums fold the input zero point, and the cube scatter is a pure
re-layout. Straightening them in place is measured in [../chips/rk3576.md](../chips/rk3576.md)
§"The host cube scatter".

### Resident weight handles

**Conditions:** MobileNetV1-224, H96 MAX M9, 2026-07-29. Times are per inference. The per-call
and resident figures are paired on one build, measured after the straightening in
[../chips/rk3576.md](../chips/rk3576.md) §"The host cube scatter".

**Not characterized:** the difference between this per-call figure, 98.8 ms, and the 115 ms the
straightening measured earlier the same day.

`rocket_conv2d_int8_pack_rk3576()` returns a handle a caller packs once and passes per
inference. The two weight loops and the BO allocation and release they need are 65% of the
inference time. Taking them out of the inference moves it 98.8 ms -> 33.6 ms, a 2.94x speedup on
the same 40 submits [HW sweep, H96 MAX M9]. The result is bit-identical to the per-call entry's.
The graph's TFLite distance is the same 4792 of 5044713 elements. Each of the 29 compute layers stays bit-exact
against a CPU model of the SoC's arithmetic, and the whole top-5 is unchanged.

A transient call builds the same handle and packs each output-channel tile as its tile loop
reaches it. The per-call and resident entries are therefore one code path, with one arithmetic
[HW sweep, H96 MAX M9].

Only 29 ms of the 65 is the weight arithmetic the profile predicts. The BO teardown runs after
the profiler's last bucket. A profile not reconciled against the measured inference time
therefore attributes 16 ms per inference to nothing at all:

| bucket | per-call | resident | removed |
|---|---|---|---|
| filter sums | 4.0 | 0.0 | 4.0 |
| feature pack | 27.8 | 11.6 | **16.2** |
| weight cube | 24.0 | 0.0 | 24.0 |
| coefficients | 0.6 | 0.0 | 0.6 |
| plan + gen | 6.7 | 2.2 | 4.5 |
| submit | 7.6 | 7.6 | 0.0 |
| de-tile | 6.6 | 6.8 | 0.0 |
| **BO teardown** | **16.1** | 0.0 | **16.1** |
| library total | 93.3 | 28.3 | 65.0 |

Three entries in that table are buffer costs. Unmapping the BOs a layer touched is 16 ms of the
98.8 ms per-call graph: a megabyte-sized dirty mapping per tile, once per layer per inference. A
profile that ends at the de-tile does not show it.

The feature bucket more than halves with its scatter loop unchanged. Of its 28 ms, 16 are the
allocation, the zeroing memset and the two cache-maintenance ioctls. Most of the stem's feature
pack is therefore the buffer, not the scatter. The `plan+gen` bucket is mostly an allocation
too, the regcmd BO. The scatter that remains, 11.6 ms, is the transpose itself, which the next
section vectorizes.

With resident handles the graph takes 33.6 ms, and the submit is 22% of the library time rather
than 8%. The resident path is therefore much closer to dispatch-bound. What is left on the host
is the feature scatter (11.6), the de-tile (6.8) and the 2.2 ms the table calls `plan + gen`.

#### Sentinel stamp cost

The 2.2 ms of `plan + gen` is not register-program generation. A resident handle's DMA addresses
do not move between calls, so its whole per-row-task stream depends only on state frozen at pack
time. `librocketnpu` generates the stream once, at pack time, bit-identically. Doing so saves
0.13 ms of 33.6 [HW sweep, H96 MAX M9, 2026-07-29].

What the bucket holds is the output's sentinel stamp. The stamp writes a sentinel value over the
output before the job, so that a task that wrote nothing is detectable. The bucket is 2.06 ms per
inference with the stamp on and 0.02 ms with it off. Neither the row plan nor the emitter is
therefore measurable at all. Cutting the stamp's `memset` from the whole output to 64 bytes
moves the bucket 1.53 -> 1.45 [HW sweep, H96 MAX M9, 2026-07-29].

> [!CAUTION]
> A convolution can inherit the wide-output hazard from whatever job ran before it. It then
> writes nothing and reports no error. Software must stamp each output with a sentinel before
> the job, and check it after [HW sweep, H96 MAX M9, 2026-07-29].

The stamp is therefore required, and the only optimization is where it is written. The library
writes the next call's stamp inside the de-tile's own bracket. The output is already inside a
`PREP_BO`/`FINI_BO` pair there, so a resident tile makes one round trip per call instead of two.
The fold saves about 0.5 ms of 30, and never costs time [HW sweep, H96 MAX M9].

Removing one of the stamp's two brackets does not halve its cost. The whole stamp costs 1.7 ms
with two brackets and 1.2 with the fold. The fill's own cache maintenance moves with it into
whichever bracket contains it, because `dma_sync_sgtable_for_device` walks the whole buffer
however much of it is dirty. For the same reason the 64-byte memset saves almost nothing. What
the fold removes is one ioctl pair per tile per call, and that is the 0.5 ms.

> [!NOTE]
> Under `ROCKET_RK3576_INT8_PROF=1` the `plan+gen` and `de-scatter` buckets are not additive
> with the inference time for a change this small. The stamp also warms the output for the
> de-tile that follows it, and that saving appears in a different bucket from its cost.
> Measure such a change with long interleaved runs of the whole graph [HW sweep, H96 MAX M9,
> 2026-07-29].

### The CHW <-> NC1HWC2 transpose

The host converts between a row-major tensor and the cube with a 16×16 block transpose. The cube
interleaves sixteen channels into every sixteen-byte atom, so a row-major tensor and a cube are a
transpose and not a copy. Written an element at a time, the transpose moves one useful byte per
destination cache line. The sixteen channels of a group each walk the same lines again, four
line touches per byte moved. A 16×16 block does the same work in whole vectors and touches every
line once. It makes sixteen 16-byte loads and four levels of `TRN1`/`TRN2`, and stores four full
cache lines.

The block must keep at most eight vectors live. A 16×16 written as one array of sixteen
`uint8x16_t` needs thirty-two registers with its scratch, and the compiler then puts the array
on the stack. On the RK3576 that form measures 3x slower than the same transpose done as two 8×8
halves plus a swap of the off-diagonal quadrants. The quadrant form never has more than eight
vectors live. Against the scalar loop, at `ic=128 px=3136`:

| operation | scalar loop | array of sixteen | quadrants |
|---|---:|---:|---:|
| pack | 0.42 | 0.23 | **0.078** |
| unpack | 0.35 | 0.21 | **0.095** |

The full-group predicate is a constant argument to one `always_inline` helper. The compiler
therefore emits the unpredicated sixteen-channel loop and the partial one from one source. A
per-lane runtime predicate costs 10-25%. The pack still writes the dead lanes: they hold the
cube's channel padding, which must be zero, and a whole atom is stored either way.

On the graph the vectorized transpose is 1.41x. The result is bit-identical, with the same
4792-element TFLite distance and the same top-5:

| | before | after |
|---|---|---|
| feature scatter | 10.91 | 4.94 |
| de-tile (readback) | 7.90 | 5.30 |
| submit | 7.17 | 7.22 |
| library total | 26.26 | 17.70 |
| **inference** | **29.5-30.3** | **20.3-21.3** |

With the vectorized transpose, the submit is the largest single bucket, at 41% of the library
time.

What is left in the two transpose buckets is the bytes, not the loop. The de-tile's `PREP_BO`,
the DPU write drain, measures only 0.53 ms of the 5.3, and the transpose is the rest. At ~5 MB
each way the two buckets run within about 2x of the RK3576's single-thread `memcpy` rate
([../chips/rk3576.md](../chips/rk3576.md) §"DRAM bandwidth"), against 6-9x before the block
transpose. They are therefore within about 2x of the cost of moving the bytes at all, and
further vectorization can recover little.

## Cube layout between consecutive layers

**Conditions:** H96 MAX M9. The stride sweep covers +3, +16 and +64 elements over five
geometries, at one row task and at ten.

**Not characterized:** a task-window stride (`0x1098`) other than its derived value.

A producer writes its output at its own channel-group stride: `ow*oh` on the direct path and
`round4(ow*oh)` on the depthwise path ([../chips/rk3576-regcmd.md §"The depthwise output surface
stride"](../chips/rk3576-regcmd.md#the-depthwise-output-surface-stride)). A packed feature
cube's channel-group stride is `ih*iw`. The CNA reads a feature cube's channel groups at the
stride in `0x1094`, which the regcmd encoder names `R76_CNA_SURF_FULL`. It honors any value at
or above the plane ([../chips/rk3576-regcmd.md §"Window
addressing"](../chips/rk3576-regcmd.md#window-addressing)) [HW sweep, H96 MAX M9].

> [!CAUTION]
> A consumer whose `0x1094` does not match its producer's channel-group stride reads the wrong
> bytes as each group. It produces incorrect output and reports no error. Software must program
> the consumer's `0x1094` with the producer's stride.

Whenever the plane and the channel rounding agree, layer n's output is layer n+1's feature
cube, byte for byte. The transposes between two NPU layers can then be skipped.

In `librocketnpu`, both ends of a join are properties of the resident handle, which already has
the input and output BOs and a fixed fd. Three calls make the join:

- `rocket_conv2d_int8_cube_of_rk3576()` describes a handle's output.
- `_cube_in_rk3576()` makes another handle read from that output instead of scattering.
- `_cube_out_rk3576()` leaves the output where the DPU wrote it.

`in` and `out` can be NULL for an end the handle does not use. A cube-in call does no cache
maintenance either. Nothing on the CPU touches those lines, so the scatter's
`PREP_BO`/`FINI_BO` pair goes with the scatter.

`rocket_rk3576_out_surf_elems` pads the stride only in the depthwise case. The descriptor
`rocket_rk3576_cube` contains its producer's stride, and `_cube_in_rk3576()` passes it to the
program. A depthwise producer at a plane whose element count is not a multiple of four
therefore joins like any other. The library refuses a stride shorter than the plane, which makes
the groups overlap. No producer in the library makes one.

The library refuses a join in three cases, each a shape where the two buffers are not the same
object:

- More than one output-channel tile.
- An input channel count that is not a multiple of 32, from a producer that does not say
  what its tail holds.
- A mismatched fd or plane.

Each output-channel tile has its own output BO, and several BOs are not one cube. Nothing in a
MobileNet reaches the tile bound. A 1×1 layer's weight slice is 32 KiB, against the 144 KiB
that leaves the tile at the whole layer.

For the channel-count bound, a handle's own cube fills the channels past `ic` with its border
constant. A producer's output holds whatever the DPU wrote into its own padded output channels.
Their weights are zero, so nothing reaches the MAC either way. What reads them is the
coefficient group's B term, which sums every tap of the whole programmed group.

The tail is harmless to the B term in two cases. A symmetric weight quantization has no B term
at all. A direct producer's partial output group holds its output zero point, and at a graph
join that is the consumer's own border constant. The descriptor `rocket_rk3576_cube` has
`pad_from`/`pad_value`, so that the tail is stated rather than assumed, and the library checks
it against the consumer's `in_zp`. A depthwise producer's output has the deeper tail the host
fills at pack time. See "What a producer leaves past its live output channels" below.

A cube can also be a slice of a larger buffer, which is what a residual topology needs. See
"Concatenation at a cube's base address" below.

### What a producer leaves past its live output channels

**Conditions:** H96 MAX M9, 2026-07-31. Direct producers at `oc` 16, 24 and 144 and `k` 1 and 3.
Depthwise producers at `oc` 16, 48 and 144.

**Not characterized:** whether the DPU leaves alone the unaddressed channels of a group it
writes.

A consumer whose `ic` is not a multiple of 32 reads the channel count rounded up to 32. The
join therefore turns on the channels between the producer's `oc` and that boundary, and their
content is known.

A direct producer's partial output group holds its output zero point. The channels past `oc`
are packed with zero weights and a zero A term, so their accumulator is zero. The OUT_CVT then
sends them to the output zero point. At a graph join that is the consumer's input zero point,
because the two are one tensor's quantization. It is therefore exactly the border constant the
consumer's own cube would have filled [HW sweep, H96 MAX M9, 2026-07-31].

> [!CAUTION]
> A padding channel packed with a B term does not hold the output zero point. The term `B` is
> per output channel, and a packer loop over the programmed count gives every channel one. Such
> a padding channel holds `requant(B*sum(x))`, which depends on the data. A consumer that reads
> the tail produces incorrect output with no error, while the producer's live channels are
> unchanged. Software must pack every channel past `oc` with no B term.

> [!NOTE]
> A test whose producers all use symmetric weight quantization has no B term, so it cannot
> detect a padding channel that was given one. Isolated per-op tests cannot detect it either.
> Test a join with an asymmetric-weight producer.

A depthwise producer is programmed with the raw output count, and the DPU writes nothing past
it. The padding elements still hold the sentinel afterward [HW sweep, H96 MAX M9, 2026-07-31].
Its output is therefore a group short of what a consumer reads. At `oc=144` it holds 9 groups,
against the 10 that a direct consumer of 144 channels walks.

> [!CAUTION]
> A consumer reads the channel count rounded up to 32, so behind a depthwise producer it reads a
> group the DPU never wrote. Software must allocate that group and fill it with the output zero
> point.

`librocketnpu` does this at pack time, for a depthwise producer that writes its own output
buffer. Its tile allocates the deeper output and fills the group past its own with the output
zero point once. Nothing overwrites that fill. The program is told the raw count and does not
address those bytes, and the sentinel covers the written groups alone. Read back after an
inference, the fill still stands [HW sweep, H96 MAX M9, 2026-07-31]. Such a producer therefore
declares the same tail a direct one does, and the join is available.

A depthwise producer placed into a caller's slice by `rocket_conv2d_int8_cube_out_at_rk3576()`
declares no tail past the groups it writes. Nothing fills the group past them there.

Where the raw count does not fill its last written group, the join is refused. At `oc=24`
(two groups holding 32 channels), the tail lies inside a group the program does write, so the
output stays short [HW sweep, H96 MAX M9, 2026-07-31].

### Join pass

**Conditions:** MobileNetV1-224, H96 MAX M9, 2026-07-29.

Software must decide the cube joins in a pre-pass over already-packed handles, not as the graph
runs. The producer's "leave it in the cube" and the consumer's "read it from there" are one
decision. A producer that skipped its de-tile for a consumer that then refused the cube
leaves no row-major tensor at all.

On MobileNetV1-224 the pass makes 24 joins of 30 layers. It runs with the asymmetric-pad
lowering below, which closes four of the joins. Together they take the graph
20.9-21.4 -> 11.9-12.3 ms, 1.73x, on the same 40 submits and 40 tasks. The two together save more
than the sum of their separate savings [HW sweep, H96 MAX M9, 2026-07-29]:

| asymmetric pad | cube chain | ms per inference |
|---|---|---|
| materialized | off | 20.9-21.4 |
| extent | off | 20.1-20.5 |
| materialized | on | 15.7-16.4 |
| **extent** | **on** | **11.9-12.3** |

The profile buckets before and after:

| bucket | before | after |
|---|---|---|
| submit | 8.4 | 7.5 |
| feature scatter | 6.5 | 3.0 |
| de-tile | 5.5 | 0.8 |
| plan+gen (includes the stamp) | 0.8 | 2.6 |

Joining leaves the arithmetic unchanged. Every layer still materialized is identical to the
row-major run, and the logits are byte-identical. The top-5 is unchanged, and the 4792-element
TFLite distance does not move [HW sweep, H96 MAX M9, 2026-07-29].

> [!NOTE]
> A wrong byte inside a join reaches the next materialized layer. Test a joined graph against
> its row-major run, at every layer the joined graph still materializes and at the final output.

On the joined graph the submit is 62% of the library time: 40 submits of about 190 us each.
They are well under the ~340 us per-submit overhead, so the optimization there is the submit
count across layers. A submit removed saves about 110 us on this graph. Chaining a tile's row
tasks into one hardware kick takes the graph from 40 submits to 29 and from 11.8-12.2 ms to
10.5-11.1. The 40 tasks are the same in both configurations [HW sweep, H96 MAX M9, 2026-07-29].

The saving is 9-11% here against about 3% over the per-op convolution tests, because a graph's
layers are cheap enough for the submit to dominate. Chaining a tile's row tasks is therefore on
by default, and `ROCKET_RK3576_BATCH_TASKS=0` turns it off. Chaining across layers extends the
same optimization and saves much more (see below) [HW sweep, H96 MAX M9, 2026-07-29].

The sentinel stamp is 0.9-1.9 ms of the 12 on either path. On the row-major path it is the fill
inside the de-tile's bracket. On the cube path it is its own `PREP_BO`/`FINI_BO` pair at the
start of the next call. The stamp goes there because the output is read as an input before that
handle runs again. The two cost the same, so the cube chain does not make the guard more
expensive [HW sweep, H96 MAX M9, 2026-07-29].

## The write guard on a graph

The write guard stamps each output with a sentinel before a submit, and checks after it that
each task wrote. Once a chained run of layers is one kick, the guard is the largest single
term of a graph's wall-clock time. The sections below give its redo rule, its cost and the
changes that reduce it.

### Redo of a retired job

The `rocket` 1.6.0 driver retires a job 125 ms after its kick and signals the fence as if the job
had completed [source-confirmed, patch `rk3576/npu/0025`]. A job that raises no completion meets
it ([../chips/rk3576.md](../chips/rk3576.md) §"int8 matmul jobs that raise no completion").

> [!CAUTION]
> A retired job can leave a partial output, which a check for one written byte per task passes.
> Software must time each job, from before the submit ioctl to the fence, and redo a job that
> reaches the backstop.

The kick follows the submit's start, so a retired job measures at least 125 ms. A healthy job
measures less, because the backstop bounds it too. Timing cannot detect a job the kernel retires
early, under a DPU grace shorter than the drain, which measures fast. It also cannot detect a
partial write by a job that completed. A job queued behind another process's job measures long
as well, and costs one redo.

The library redoes a job at or past the backstop after a power cycle, like a job that wrote
nothing. The entry refuses once its attempts run out. The int8 matmul plans its tiles so that no
job stalls. If a partly written job does stall, the int8 matmul re-plans it narrower. Where the
write check is one byte per task, the task extents go back to the sentinel before the redo. A
redo that writes nothing then cannot pass on the retired attempt's bytes.

The counter `rocket_rk3576_retired_counts()` counts the scored retirements, split by whether the
write check alone passed the output. The hardware profile's `backstop_ms` holds the value, and
`ROCKET_RK3576_BACKSTOP_US` overrides it.

The guard sites are:

- The conv submit helper, which covers the direct, depthwise, packed-image and fp16
  convolutions.
- The chained kick.
- The int8, int32 and fp16 matmul writers.
- The pool.

On every test that submits through a library entry, the scored retirements equal the kernel's
own count of retired jobs. At five stall `K` where a write check alone returned incorrect
output, the entry's output is correct [HW sweep, H96 MAX M9, `rocket` 1.6.0, 2026-09-27].

### Sentinel guard brackets

The `PREP_BO` ioctl syncs a whole BO for the CPU, and `FINI_BO` syncs the whole BO back for the
device. Neither takes a range, so a bracket costs two passes over the BO however few bytes are
touched inside it.

The sentinel guard costs a bracket per output per call, and the bracket count is what software
can reduce. Once a chained run of layers is one hardware kick, the guard is the largest single bucket in
the whole graph. Setting `ROCKET_RK3576_I32_SENTINEL=0` takes MobileNetV2 from 10.0 to 5.9 ms
and MobileNetV1 from 6.4 to 3.5. At the default guard the memsets are 17-27% of the guard, and
the cache maintenance is the rest ("Fill and bracket terms of the guard").

Unfolded, the chained path opens two brackets on every layer's output per call, for a single
CPU visit. One checks whether the layer's tasks wrote, and one puts the next call's sentinel
back. Folding the re-stamp into the verify bracket is 10.0 -> 8.8 ms on V2 and 6.4 -> 5.4 on V1.
Both configurations are byte-identical and run the same submits and tasks. The order is safe
because the redo path re-stamps every output anyway, so a layer stamped before a later one is
found bad loses nothing.

Two outputs keep separate handling, because something outside the kick reads them after it. The
last layer keeps its own handling under the de-tile, which must read the output before
anything overwrites it. An output shared with something outside the chain still takes its stamp
at the start of the next call.

The remaining cost is one bracket per BO per call. It is proportional to bytes rather than to
ioctls [HW sweep, H96 MAX M9, 2026-07-31].

### One bracket per BO

One bracket per BO, rather than per layer, reduces the guard without reducing what it checks,
unlike the two optimizations below. A bracket syncs the whole object. Whenever a concatenation
is placed, every operand of it writes a slice of one buffer, so several layers of one chained run write
that buffer. A per-layer pass therefore re-syncs it once per operand, checking a different slice
each time. Naming each BO once checks exactly the same bytes.

Software must key the set of open BOs on the kernel handle. A placed slice has its own
descriptor of the same underlying object. Keyed on the `rocket_bo` pointer, the set finds no
shared BO on any graph, including one with fourteen placed concatenations.

**Conditions:** H96 MAX M9, CPU governor `ondemand`, 2026-08-03.

Both configurations are byte-identical per layer and return TFLite's own top-1 on the same
submits and tasks [HW sweep, H96 MAX M9, 2026-08-03]:

| | brackets, per layer | bytes | guard | time | brackets, per BO | bytes | guard | time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV1 | 30.3 | 4987 KiB | 2.03 | 5.0 | 30.3 | 4987 | 2.07 | 5.0 |
| MobileNetV2 | 85.4 | 8665 | 3.36 | 6.9 | **65.4** | **7721** | **3.19** | **6.6** |
| ResNet-18 | 51.1 | 8243 | 2.87 | 7.4 | **35.1** | **5303** | **2.10** | **6.5-6.6** |
| Inception V1 | 122.2 | 14465 | 4.75 | 8.8 | **68.2** | **7207** | **2.83** | **7.0-7.1** |
| Inception V3 | 122.4 | 19713 | 6.35 | 42.3 | **102.4** | **17272** | **5.80** | **40.7-41.9** |

MobileNetV1 has no placed concatenation and shares no BO, and its figures do not change. The guard-bucket
saving equals the inference-time saving to 0.05 ms on every graph. The saving is therefore the
bracket bytes and nothing else.

Inception V3's row was measured before its fifteenth concatenation was placed. With it placed,
that graph takes 38.4-38.8 ms.

Ranged brackets stay per layer. Two slices' ranges are disjoint, so there is nothing redundant
to remove, and a pooled bracket would have to cover their union.

Naming each BO once adds a linear scan of the open set per layer, bounded by a chained run's distinct BO
count. It also adds one pointer array per chain.

### Bracket cost

**Conditions:** H96 MAX M9, BO sizes 4 KiB to 4 MiB, the median at each size, 2026-08-02.

A bracket costs a fixed 2.24 us plus 0.194 us/KiB, so its cost is in the bytes [HW sweep, H96
MAX M9, 2026-08-02]:

| | fixed cost | per KiB | rate |
|---|---:|---:|---:|
| `PREP_BO` | 1.33 us | 0.0969 us/KiB | 10.57 GB/s |
| `FINI_BO` | 1.11 us | 0.0970 us/KiB | 10.56 GB/s |
| one bracket | 2.24 us | 0.1939 us/KiB | 5.28 GB/s |

An ioctl the kernel refuses at handle lookup, with no fence wait and no cache walk, costs
1.17 us. Its cost is the syscall, the DRM dispatch and the GEM lookup. Past a few KiB, the
ioctl's cost is therefore entirely the cache walk. The memset inside a bracket adds a further 0.113 us/KiB. It
is charged once per output per call rather than once per bracket. Each output is stamped either
in the verify bracket or in the next call's stamp loop, and never both.

The chain counts its own guard: `ROCKET_RK3576_INT8_PROF=1` prints brackets and bytes, which
turns the guard's milliseconds into arithmetic. Brackets times the fixed cost is 0.07-0.27 ms,
4-7% of each graph's guard, so the fixed cost is not the term to remove.

### Fill and bracket terms of the guard

**Conditions:** H96 MAX M9, CPU governor `ondemand`, `INT8_PROF=1`, 2026-08-03.

The guard is a fill and a bracket, and the fill is 17-27% of it. The fill is the memset that
puts the sentinel back. It is charged per layer, because every slice of a shared buffer still
has to be filled. The rest is the two cache-maintenance ioctls, the scan and the loop. Naming
each BO once reaches only that second term. `ROCKET_RK3576_INT8_PROF=1` prints the two
separately (`write guard = fill ... + brackets and scan ...`).

The table gives the default guard per inference [HW sweep, H96 MAX M9, 2026-08-03]:

| | guard | fill | fill bytes | rate | brackets + scan |
|---|---:|---:|---:|---:|---:|
| MobileNetV1 | 2.07 | 0.56 (27%) | 4933 KiB | 0.113 us/KiB | 1.52 |
| MobileNetV2 | 3.19 | 0.81 (25%) | 7116 | 0.114 | 2.38 |
| ResNet-18 | 2.12 | 0.40 (19%) | 3470 | 0.114 | 1.73 |
| Inception V1 | 2.79 | 0.53 (19%) | 4581 | 0.116 | 2.26 |
| Inception V3 | 5.80 | 1.00 (17%) | 8629 | 0.117 | 4.82 |

The rate is flat at 0.113-0.117 us/KiB on all five graphs, the 0.113 that "Bracket cost"
measures for a memset inside a bracket. The fill is therefore a memset running at the SoC's
memset speed, and the loop around it has nothing to remove.

The fill cannot be reduced without reducing what the check covers. The check reads the whole
extent on a task that wrote nothing, so the stamp has to cover whatever the check can read.
Narrowing the fill is narrowing the check. Naming each BO once reduces the guard while keeping
the check whole, and it is the default.

### Narrowing's fill and bracket shares

**Conditions:** H96 MAX M9, CPU governor `ondemand`, `INT8_PROF=1`, 2026-08-03.

`GUARD_NARROW=1` narrows the guard's footprint ("Narrowed footprint and ranged ioctl", below).
The table sets it against the deduplicated default, on the same chained-run structure. All five graphs
pass and are byte-identical, with TFLite's own top-1 [HW sweep, H96 MAX M9, 2026-08-03]:

| | guard | narrowed | saving | of which fill | of which brackets + scan |
|---|---:|---:|---:|---:|---:|
| MobileNetV1 | 2.07 | 0.67 | 1.40 | 0.53 (38%) | 0.88 |
| MobileNetV2 | 3.19 | 1.20 | 1.99 | 0.75 (38%) | 1.23 |
| ResNet-18 | 2.12 | 0.71 | 1.41 | 0.38 (27%) | 1.04 |
| Inception V1 | 2.79 | 1.50 | 1.29 | 0.49 (38%) | 0.80 |
| Inception V3 | 5.80 | 1.60 | 4.20 | 0.97 (23%) | 3.27 |

The fill is 23-38% of what narrowing saves. The rest is the bracket bytes, which need the
ranged ioctl. Narrowing saves 1.3-4.2 ms across the five graphs.

The bracket count rises as the footprint narrows, because ranged brackets are not pooled: two
slices' ranges are disjoint. The bracket counts, default against narrowed:

| | default | narrowed |
|---|---:|---:|
| MobileNetV1 | 30 | 30 |
| MobileNetV2 | 65 | 85 |
| ResNet-18 | 35 | 51 |
| Inception V1 | 68 | 122 |
| Inception V3 | 102 | 122 |

Inception V3's narrowed guard is 1.60 ms, where brackets times the fixed cost predicts 0.30.
About 1.3 ms of it is therefore per-range kernel work, the same residual this sheet records
elsewhere, at a few hundred ns per 64-byte range.

### Narrowed footprint and ranged ioctl

The bytes the write guard touches fall with two changes made together: a narrowed footprint and
a ranged cache-maintenance ioctl. The check asks whether a task wrote anything, so it reads only
up to the first byte that differs from the sentinel. The stamp has to cover whatever the check
can read, and on a task that wrote nothing that is the whole extent. With
`ROCKET_RK3576_GUARD_NARROW=1`, the stamp and the check each cover one cache line per (task,
channel group), tens of bytes rather than the whole output.

Narrowing alone removes only the fill, because `PREP_BO` and `FINI_BO` sync the whole BO
however few bytes the bracket touches. The ranged ioctls `DRM_ROCKET_PREP_BO_RANGES` and
`_FINI_BO_RANGES` (`patches/rk3576/npu/0023`, interface 1.5) take an array of
`{offset, size}` and sync only those ranges. The two changes are conjunctive. A ranged ioctl
names a small footprint only where the guard has one, and the union of the un-narrowed
extents is the whole output.

With both changes, every graph runs on the same submits and tasks, byte-identical per layer,
and returns TFLite's top-1 [HW sweep, H96 MAX M9, 2026-08-02].

#### Guard forms compared

The three guard forms compare as follows, against the guard off, in ms per inference
[HW sweep, H96 MAX M9, 2026-08-03]:

| | default | guard off | per-kick | narrowed |
|---|---:|---:|---:|---:|
| MobileNetV1 | 4.9 | 3.0 | **3.0** | 3.7 |
| MobileNetV2 | 6.5 | 3.5 | **3.5** | 4.7 |
| ResNet-18 | 6.6 | 4.5 | **4.5** | 5.2 |
| Inception V1 | 7.0 | 4.2 | **4.1** | 5.7 |
| Inception V3 | 38.5-38.8 | 31.9-32.9 | 36.1-36.5 | 34.9-35.5 |

**Conditions:** H96 MAX M9, CPU governor `ondemand`, the default guard naming each BO once.

Inception V3's default column predates scoring the tail with the divisor-lag check. With that
check, its default is 33.4-33.7 ms. The guard deltas are a separate term, and they are not
expected to move with it [expected].

The per-kick form, `ROCKET_RK3576_GUARD_PER_KICK=1`, checks one output per kick, and the
default checks every layer's. Checking fewer outputs removes nearly all of the guard's cost.
On the four graphs that run as one kick, the per-kick form matches the guard-off time. It runs
on the same submits and tasks, with the logits and every materialized layer byte-identical.

On Inception V3 the per-kick form removes only 2.3-2.4 ms of the 5.9-6.6 ms guard. Inception V3
runs its convolution calls outside any kick in the twenties, and a per-kick guard checks none of
them. The per-kick guard's coverage therefore depends on the graph's chained-run structure.

#### Per-kick coverage of the wide-output hazard

One check per kick is enough because the hazard is per submit. A job writing a DPU output
element wider than one byte through a partial output stage leaves the next submit, of any kind,
unwritten (RK3576-E2). A cross-layer kick is
one submit, so every output in it detects that hazard equally. With every layer checked or with
only the last output checked, every inference that follows such a job returns its correct
result. With the guard off, half of them do not [HW sweep, H96 MAX M9, 7.2.3-1, `rocket` 1.6.0,
2026-09-24].

**Not characterized:** a program that fails in the middle of a stream while the last program
still writes. The full per-layer guard targets that failure, and one check per kick cannot
detect it.

#### Guard forms with the governor pinned

With the CPU governor pinned to `performance`, the two cheaper forms save less than the
`ondemand` table shows. Under `performance`, the configurations' differing host work cannot
move the CPU clock. On Inception V3 [HW sweep, H96 MAX M9, 2026-08-04]:

| configuration | ms | vs default |
|---|---:|---:|
| default | 29.53 | n/a |
| `GUARD_PER_KICK` | 27.53 | −2.00 |
| `GUARD_NARROW` | 27.20 | −2.33 |
| `POOL_LAG_TAIL` | 26.00 | −3.53 |
| per-kick + tail | 24.07 | −5.46 |
| conv write guard off | 24.50 | −5.03 |

**Conditions:** H96 MAX M9, Inception V3, CPU governor `performance`, spread within a
configuration 0.3 ms.

The per-kick guard and the lag-check change add to within 0.07 ms (−2.00 + −3.53 against −5.46
measured), so they are separate terms. Per-kick plus tail runs faster than the conv write guard
off, while still checking every kick. Turning the conv write guard off removes only that guard,
and the combination also removes the lag check's cost.

> [!NOTE]
> Under `ondemand`, the combination cannot be told from the tail change alone: 33.50 ms against
> 33.43, at a 4.7 ms spread. Pin the CPU governor to `performance` to compare configurations
> that differ in host work.

#### Coverage and cost of the narrowed check

Narrowing reduces the check's coverage, and it is off by default. One line per channel group per
task still detects the hazard the guard exists for. A wide-output job prevents the whole next
submit from writing, so every line of every extent still holds the sentinel. The narrowed check
misses a partial write that spares exactly those lines, the same class of write that
`GUARD_PER_KICK` leaves unchecked.

Narrowing also raises the false-alarm rate. A task whose first 64 bytes per group genuinely equal
the sentinel byte reads as unwritten and costs a redo. The full check reads a task as unwritten
only when its whole extent equals the sentinel.

On Inception V1, narrowing saves 1.2-1.3 ms over the default, which names each BO once, and 3.1
over a default with a bracket per layer. Naming each BO once therefore gives more than half of
narrowing's saving over the per-layer default, with no coverage trade.

The narrowed guard's remainder is well above the bracket count times the fixed per-bracket cost
(0.59 against 0.07 ms on V1, with a bracket per layer, 2026-08-02), because each range costs
kernel work of its own. The work is `memdup_user`, the validation pass, and a scatterlist walk
plus an `iommu_iova_to_phys` per range. A ranged bracket costs 12-20 us against the 2.24 us fixed
cost, and contains tens of 64-byte ranges. The work is therefore a few hundred nanoseconds a
range. The per-range work is what the ranged ioctl adds. On the four graphs measured that day, it
takes back a fifth to a third of the saving a fixed-cost-only model predicts [host-computed].

> [!NOTE]
> A test that feeds the same input on every iteration cannot detect a stale output. A layer
> whose program did not run leaves its output holding the previous inference's contents. On a
> repeated input those are the contents it would have written, so the graph returns the right
> answer and every check downstream passes. Vary the input to test for staleness.

## Cross-layer chaining in one kick

A chained stream honors read-after-write between its programs [HW sweep, H96 MAX M9]. The
ordering holds across layers, although `PC_DONE` fires before the DPU's writes become visible
([../chips/rk3576.md](../chips/rk3576.md) §"Retirement on the DPU completion"). Two 1×1 int8
convolutions, the second reading the first's output cube, give the same result in one batched
job as in two submits. The planes run from 2 KB to 387 KB. One tile's row tasks write disjoint
rows, so chaining them does not depend on this ordering.

A chained run of consecutive cube-joined layers can therefore run as one job. Between two joined layers
there is no host work at all: no de-tile, no scatter and no cache maintenance. The cube join
between two layers is the eligibility test for chaining them.

`librocketnpu`'s `rocket_conv2d_int8_chain_new_rk3576()` asserts the joins against the BO
addresses rather than the flags. It plans and generates every program once, because the stream
is a function of the frozen geometry and is not regenerated per call. It lays the stream out
contiguously in one BO, with each trailer linked to the next.

On MobileNetV1-224 with the transposes removed, the submit is the largest single cost: 29
submits for 29 layers, each ~190 us against 130-250 us of compute. Chaining takes the graph from
10.4-10.5 ms to 6.7 ms and from 29 submits to 6, on the same 40 tasks. The logits are
byte-identical, and every materialized layer is identical. Two kicks contain 25 of the 30 layers.

Timed per layer, the 23-layer kick is then 65% of wall-clock time, and no layer outside it
reaches 0.31 ms. The PPU pooling layer is another 20% in that configuration, and 2% after the
changes described in "The pooling layer" below.

### Cost inside a kick

Inside a kick, the cost is front-loaded on the stem. The kick is timed as one bucket, with its
chained run capped at `m` layers by `ROCKET_RK3576_NET_KICK_MAX=<m>`. The layers the cap excludes leave
the kick and become their own buckets. The bucket at cap `m` is therefore the cost of the chained run's
first `m` programs. The difference between consecutive caps is one program's cost inside the
kick [HW sweep, H96 MAX M9, 2026-07-31]:

| cap | 2 | 5 | 10 | 15 | 20 | 23 |
|---|---|---|---|---|---|---|
| kick bucket, ms | 1.78 | 2.53 | 3.25 | 3.52 | 3.77 | 3.98 |
| graph, ms | 8.4 | 7.0 | 6.7 | 6.0 | 5.3 | 4.9 |
| submits | 27 | 24 | 19 | 14 | 9 | 6 |

The first two programs are 1.78 ms of the 3.98, 45% of the kick, and the other 21 cost 0.105 ms
each. The two are the 224×224 stem and the depthwise that follows it, by far the largest planes
in the network. The kick's cost is the stem's compute plus a fixed per-program cost. A program
costs ~105 us inside a chained stream against ~270 us as its own submit. The graph falls
0.167 ms for each layer the cap moves into the kick.

Both figures are under the ~340 us per-submit overhead of a driver round trip. That overhead
bounds a round trip that waits for a full output to drain, not one program. Of the 23
programs, 21 are at the fixed per-program cost, which chaining cannot lower. The stem is the
remaining cost inside the kick, and its optimization is off by default.

> [!NOTE]
> A kick's bucket is the whole round trip of `PREP`, submit, verify, re-stamp and de-tile,
> charged to the chained run's first layer. Compare it against the guard turned off before reading a
> finite difference as a program's cost.

On MobileNetV2, most of a kick's bucket is not program execution. The bucket over its three large
chained runs measures 84% of a 10.0 ms wall-clock time. The finite difference puts most of chained run 1 in its
first four layers, the largest planes in the network. Turning the sentinel off moves the same
three buckets [HW sweep, H96 MAX M9, 2026-07-31]:

| MobileNetV2 chained run | 1 | 2 | 3 |
|---|---|---|---|
| kick bucket, sentinel on, ms | 4.49 | 2.59 | 1.32 |
| kick bucket, sentinel off, ms | 2.57 | 1.02 | 0.91 |

Of the wall-clock time attributed to the three kicks, 41% is the guard's cache maintenance on the
outputs inside them.

### Chaining traps

Two defects in a chain produce output of the correct size with incorrect values, and report no
error.

> [!CAUTION]
> In a chained stream, both trailer fields describe the next segment. The driver programs
> `PC_BASE_ADDRESS` and `PC_REGISTER_AMOUNTS` from task 0 alone, so every later program's address
> and its length come from its predecessor's trailer. Software must write the next program's
> address and register count into each trailer. A tile's row tasks are all the same size, so a
> trailer that passes its own program's count still works there. Across layers a depthwise
> program and a direct one differ, and the mis-sized fetch leaves the affected program writing
> nothing at all.

> [!CAUTION]
> A sentinel stamp replaces the output it covers. Software must stamp in a chain's epilogue
> bracket only an output that the chain rewrites before anything reads it. The next program in
> the same kick reads every interior output, so its stamp can go in that bracket. The stamp costs
> one `PREP_BO`/`FINI_BO` pair per output per call. A last layer that leaves a cube is read later
> in the same inference, by a layer outside the chain. If it is stamped in the epilogue, its
> consumer computes on 0xA5, and nothing before the next materialized layer detects it.

### Stream length and the completion wait

Two PC signals change state before a chained stream ends. `PC_DONE` is raised per task.
`PC_OPERATION_ENABLE` is a self-clearing go bit, and reads back 0 at every poll of a running
kick ([../chips/rk3576-regcmd.md §"Per-task `PC_DONE` and the `PC_TASK_STATUS`
counter"](../chips/rk3576-regcmd.md#per-task-pc_done-and-the-pc_task_status-counter)) [HW sweep,
H96 MAX M9].

The patch `patches/rk3576/npu/0012` retires a job when the writing block raises its own
completion in `INTERRUPT_RAW_STATUS`. Below interface 1.4, without `0022`, that wait is bounded
by `dpu_grace_us`, 500 us by default and measured on a job of one task. A job that raises no
completion therefore cannot stall waiting for one. The wait starts on `PC_DONE` and on
`OPERATION_ENABLE == 0`, and the second is true from the first poll onward. In a chained stream
the wait starts at the first poll tick, so the deadline has to cover the whole remaining stream,
not a drain. Where the kernel starts its completion wait therefore
sets the length a chained stream can reach.

> [!CAUTION]
> Past the `dpu_grace_us` deadline, a kernel whose wait starts at the first poll tick signals the
> fence while the core is still writing. The caller reads an output that was never written. The
> driver then programs the next job into a core that has not finished the last one. Software must
> run long chained streams only on a kernel that anchors the wait on `PC_TASK_STATUS` (below).

The deadline each stream needs before its answer comes back bit-exact, measured by moving
`dpu_grace_us` under a cross-layer kick [HW sweep, H96 MAX M9]:

| programs (layers) | deadline needed |
|---|---|
| 34 (23) | under 200 us |
| 30 (23) | 200 to 500 us |
| 35 (24) | 650 to 700 us |
| 220 (44) | under 150 us |

The deadline needed is not monotone in the program count and cannot be predicted from the
stream. The 500 us default falls inside the measured range. Streams on the wrong side of it
fail, and those on the right side pass by chance. The deadline has to beat the host reaching the
output. A stream of cheap programs (the 220-program mix, at 32 channels over a 32×32 plane) can
therefore retire early and still read correctly, while a shorter stream of expensive ones
cannot.

The 35-program stream that fails at the default deadline is bit-exact at a longer one. It is
equally bit-exact at a coarser `poll_interval_us` (250 us and up), which delays the observation
and the deadline with it. The taint word does not move at either.

Scaling the deadline with the tasks one kick covers removes the bound in practice
[HW sweep, H96 MAX M9]:

| stream | channels | first failed at | chains, scaled deadline |
|---|---|---|---|
| uniform 3×3 | 128 | 4 | 40 deep |
| 5×5 | 64 | 6 | 24 deep |
| 3×3 | 96 | 9 | 24 deep |
| mixed direct/depthwise | | | 220 programs deep |

A scaled deadline is still a fitted number, and a stream of slower programs outruns it. One
224×224 k3 convolution at 256 channels row-splits into 75 chained tasks. The scaled deadline,
75 x 500 us, expires 37.5 ms in, with the task counter at 67 tasks started of 75. The layer's
output is then wrong [HW sweep, H96 MAX M9].

### Completion on the task-status counter

`PC_TASK_STATUS`, at `0x48`, is the live task counter. A set `PC_DONE` together with a zero
`PC_TASK_STATUS` marks the end of the whole kick, and nothing else does ([../chips/rk3576-regcmd.md
§"Per-task `PC_DONE` and the `PC_TASK_STATUS`
counter"](../chips/rk3576-regcmd.md#per-task-pc_done-and-the-pc_task_status-counter)).

The patch `patches/rk3576/npu/0022` anchors the wait there. Under it, `dpu_grace_us` becomes the
drain it was measured as, applied to the program that wrote last. A fixed backstop covers the
case where a job raises no `PC_DONE` at all. The patch raises the interface version to 1.4.
Below 1.4, `librocketnpu` caps a stream at 34 programs, and `rocket_batch_completion_tracked()`
reports which applies.

With the wait anchored on the counter, the 75-task convolution is bit-exact, 150 programs in one
kick [HW sweep, H96 MAX M9]. Every job class raises `PC_DONE` [HW sweep, H96 MAX M9]. The
classes include the pool, whose `PC_OPERATION_ENABLE` is `0x60` with no DPU stage, and the
LUT-load program at `0x18`.

Against a deadline scaled by the task count, the counter-anchored wait moves neither the graph's
time nor the per-submit overhead. With either, the graph measures 6.4-6.5 ms in 5 submits and the
per-submit overhead a median of 348 us [HW sweep, H96 MAX M9, 2026-07-31]. Both were measured
before the library changes that took the graph further. The wait ends on the DPU's own completion
either way, and the change moves only when it begins.

> [!CAUTION]
> On a kernel without patch 0022, a chained stream of slow programs hangs the board. The board
> does not recover on its own, answers no ping, and accepts no SSH connection. Recovery needs a
> physical power cycle [HW sweep, H96 MAX M9]. The hang is a job programmed into a core that has
> not finished the last one. Software must not submit a long stream of slow programs to a
> kernel below interface 1.4.

Past the deadline's bound, the NPU raises a DMA write error, which `rocket_job_irq_handler()`
takes as `WARN_ON` (taint +512). The IOMMU then reports `Enable stall request timed out` and
`FORCE_RESET command timed out`, after which every job on every path fails. The DMA write error
and the stalled IOMMU are one defect: a job programmed into a core that is still writing the
previous one. No stream length on a kernel with patch 0022 has reached it.

> [!CAUTION]
> After the DMA write error, every later job fails, including a geometry that passed minutes
> before. Software must read `/proc/sys/kernel/tainted` after every stream length it tests. If
> the value changes, reboot. A result taken after the taint word moves says nothing about the
> chain.

A wait that covers the whole stream costs a fence that is not early, and the driver cannot recover
it. The baseline is the 500 us deadline that starts at the first poll tick. Against it, a deadline
scaled by the task count takes MobileNetV1-224 from 5.0 to 6.3 ms per inference. The graph's chained runs
then cover 26 layers in two kicks rather than 25, in one fewer submit [HW sweep, H96 MAX M9].
That 1.3 ms is the first-tick deadline's fence signaled while most of the stream had not yet run,
and a trace shows it directly. On the 35-program kick the first `PC_DONE` comes at 12 us, and the
kick ends at 1921 us.

The DPU's own completion sets 10 us after the last program starts. It is the earliest safe
instant, and the counter-anchored wait retires on it at the same microsecond a deadline does.
Software must schedule any host work it overlaps with NPU work between the submit and the wait.
No driver signal legitimately arrives sooner.

### The pooling layer

The PPU pooling layer, a `1024x7x7 -> 1024x1x1` average, takes 1.24-1.48 ms of a 6.2 ms graph,
20% of wall-clock time. The work built on the three findings below takes the layer to 0.12 ms
and the graph to 5.1 ms, byte-identical. The profile flag `ROCKET_RK3576_POOL_PROF=1` splits the
layer's cost per call, in microseconds:

| bucket | us | what it is |
|---|---|---|
| the completion wait | 640 | `PREP_BO` on the output |
| the host cube scatter | 320 | row-major CHW into NC1HWC2 |
| three BO allocations | 300 | feature cube, output, regcmd |
| the BO teardown | 150 | |
| the stamp | 8 | |
| the submit ioctl | 13 | |
| the de-tile | 13 | |

#### Pool completion

A pool enables PPU and PPU_RDMA (`0x60`) and no DPU stage, so it raises no DPU completion. It
raises its own in the PPU bits of `INTERRUPT_RAW_STATUS`, and the patch
`patches/rk3576/npu/0021` (interface 1.3) adds `DRM_ROCKET_JOB_PPU_DONE` to retire a job on
them. The pool's wait is then 71 us, against 641 us on a 500 us `dpu_grace_us` deadline
([../chips/rk3576-regcmd.md §"Pool completion"](../chips/rk3576-regcmd.md#pool-completion)) [HW
sweep, H96 MAX M9, 2026-07-31].

A pooling submit that names no class waits for the fallback on every kernel
[HW sweep, H96 MAX M9, 2026-08-05]:

| `rocket` module | Wait for a pooling submit with no class |
|---|---|
| 1.5.0 release, `dpu_grace_us` 500 | The full 500 us grace, on every submit |
| 1.6.0 out-of-tree build, `dpu_grace_us` 0 | The 125 ms backstop of patch 0022, on every submit |

On a kernel at interface 1.3 or later, software must set `DRM_ROCKET_JOB_PPU_DONE` on every
pooling submit. On an older kernel, software must not set it: the submit ioctl rejects a flag
word it does not recognize, so the submit fails. The `DRM_ROCKET_JOB_NO_DPU_DONE` flag behaves
the opposite way, because a kernel that knows the field ignores it. A kernel without the
deadline needs every caller to name a class. With the flag set, the two modules agree to within
the run-to-run spread.

`librocketnpu`'s `rocket_ppu_done_supported()` reads the interface version. Every library entry
names its class. A raw submit with `flags = 0` does not.

> [!NOTE]
> On the 1.5.0 module, the missing flag costs 500 us a submit, which a test's total time hides.
> A pooling test of ~16 such submits moves 8 ms against a 2452 ms total, and 2033 ms on 1.6.0.
> Find unflagged pooling submits by listing every caller, not by timing.

**Not characterized:** the submits of the wide-output class (`NO_DPU_DONE`), a separate
population, were not checked for a missing flag.

#### Cube-in pooling and residency

`PPU_RDMA` honors its source surface stride as programmed, including a stride that is not a
multiple of four ([../chips/rk3576-regcmd.md §"PPU_RDMA
registers"](../chips/rk3576-regcmd.md#ppu_rdma-registers)) [HW sweep, H96 MAX M9, 2026-07-31]. A
pool can therefore take a direct convolution's output, at its `ow*oh` stride, as its input cube.

`librocketnpu` keeps the pool resident. The entries `rocket_pool_int8_pack_rk3576()`,
`_prepacked_rk3576()` and `_free_rk3576()` hold the three BOs and the register program across
calls. The entry `rocket_pool_int8_cube_in_rk3576()` points the handle at a producer's output,
so neither the scatter nor its cache maintenance runs. The transient `rocket_pool_int8_rk3576()`
packs, runs and frees, so a prepacked call is bit-identical to a transient one by construction.

Of the layer's remaining 0.12 ms, 71 us is the completion wait.
[HW sweep, H96 MAX M9, 2026-07-31]

#### Pool write guard

A pooling program with a unit kernel on one axis corrupts the next process's first two pooling
submits, RK3576-E9 ([Pooling state across
processes](../chips/rk3576-hazards.md#pooling-state-across-processes)).

`librocketnpu`'s pool write guard is whole-output, so a partial write would pass it, and the
measured rate of one is under 0.025% of test points. The guard asks whether any byte of the
output differs from the sentinel. The convolution path asks it of each row task's own extent.
The bound is at 95% confidence, and it is two orders below the cost of a per-row check
[HW sweep, H96 MAX M9].
Under memory load, a dropped block of atoms has been seen only on a writer whose output element
is wider than one byte. The pool shows none.

#### Pool cube output

The PPU writes its output at a channel-group stride of `round4(ow*oh)` unless `0x607C` programs
another ([../chips/rk3576.md](../chips/rk3576.md) §"PPU destination and source addressing"). The
CNA honors a channel-group jump at any value at or above the plane. A convolution therefore
reads a pool's output cube at the PPU's stride [HW sweep, H96 MAX M9, 2026-08-02].

`librocketnpu` makes a pool a cube producer through `rocket_pool_int8_cube_out_rk3576()`,
`_cube_of_` and `_cube_out_at_`. A pool can then sit between two convolutions with no host
transpose on either side. It can also write its output into a slice of a residual add's buffer,
beside another producer's. The cube handle contains the PPU's stride.

`_cube_out_at_` programs the slice's own channel-group stride, and refuses only a stride below
the plane. Both entries also refuse a channel count that is not a multiple of 16, because
pooling reduces within a channel. A partial group's tail is then whatever the input cube's
padding held, not a value the handle can declare.
[HW sweep, H96 MAX M9, 2026-08-02]

### The chained-run finder

A chained stream honors read-after-write between its programs, and that ordering sets the rule
for a chained run of layers. A pairwise finder grows a chained run while `cube_linked(h[i], h[i+1])` holds, that
is, while layer j reads layer j-1's output. That condition is sufficient, and the necessary one
is weaker. A stream needs two conditions:

- Every layer but the last leaves a cube.
- Every layer but the first reads a cube that some earlier member wrote, not necessarily the one
  immediately before it.

The rule is sound because the chain emits its programs in array order, so "an earlier layer" is
"an earlier program". It is also a strict superset of the pairwise test, so every chained run the
pairwise test admits still chains.

`librocketnpu`'s `rocket_chain_new_rk3576()` takes one chained run of already-joined handles and asserts
it. A frontend has a graph and no reason to know about chained runs. The planner
`rocket_chain_plan_rk3576()` therefore groups an array of nodes into maximal chained runs. It splits them at the
program-count bound. It is pure: it uses no hardware and writes nothing on the handles.

An unset node breaks a chained run. A caller uses one to mark a layer that is not a resident convolution
or pool. It also marks a point where a chained run must not start. A chain scatters the tensor it is handed, so a
layer whose input the caller prepares on the host must not begin one. The convolution-only
`rocket_conv2d_int8_chain_plan_rk3576()` and `_chain_new_rk3576()` are the same functions with
every node a convolution, and the two share one implementation.

A ResNet block that changes width shows the difference. Its 1×1 shortcut convolution reads the
block input three layers back and writes a slice of the add's buffer. A pairwise rule strands
each of the three between two kicks, even though no host work happens on either side of it. The
wider rule takes ResNet-18 from four chained runs covering 24 layers to one of 27 covering 29 programs,
byte-identical. It goes 11 submits -> 5 and 10.5 -> 9.9 ms. Neither MobileNet moves, because
each is already one chained run.

Requiring a producer inside the chained run, rather than only a stable buffer, is the conservative half of the
rule. An operand no member of the stream writes is a BO the job never names. The layout's BO
deduplication relies on every intermediate output being some member's output.

### A pooling layer inside the stream

A chained stream can contain a pooling program between convolutions. `PC_OPERATION_ENABLE`
differs per program, `0x1D` for a convolution and `0x60` for a pool, and the two bit sets are
disjoint. The bitmap does not partition a stream [HW sweep, H96 MAX M9, 2026-08-02]. A
`conv -> pool -> conv` sequence as one batched job gives the same two outputs as three submits:
the pool's, and the following convolution's. The geometries cover planes 16 to 110 and channels
32 and 64.

A classifier puts a pooling layer in the middle of a chained run. A chain built from convolution handles
alone breaks at a pool, and on ResNet-18 that isolates four layers:

- The stem
- The maxpool after it
- The final average pool
- The classifier

`librocketnpu` therefore describes a chained run by nodes rather than by convolution handles. Each node,
a `rocket_chain_node_rk3576`, contains exactly one of a convolution handle and a pooling handle.

A pool can only be an **interior** node. A pool that began a chained run would need the chain to scatter
into its cube. A pool that ended a chained run would need its output de-tiled. Both are the host work
the stream exists to remove. The finder therefore trims a trailing pool off a chained run and never opens
one on a pool.

A pool has two bookkeeping consequences. It names only its output in the job's BO lists, because
it has no weight cube and no coefficient group. Its output is also **never re-stamped inside the
verify bracket**, because the pooling handle keeps no "already stamped" flag for the chain to
clear. Its sentinel therefore goes on at the start of the next call, one extra bracket for at
most one layer.

All three networks then run as one hardware kick in one submit. All three are byte-identical,
with the same task counts and TFLite's top-1:

| network | wall-clock time, ms | submits |
|---|---|---|
| ResNet-18 | 9.9 -> 9.2 | 5 -> 1 |
| MobileNetV1 | 5.1 -> 4.9 | 3 -> 1 |
| MobileNetV2 | 6.8 -> 6.7 | 3 -> 1 |

Removing one submit saves about 0.13-0.14 ms on ResNet-18, 10 submits removed for 1.3-1.4 ms, by
the 11-submit graph's two readings of 2026-08-02 (10.5 and 10.6 ms). It saves 0.05-0.10 ms
on the MobileNets (2 removed for 0.1-0.2). With tens of submits still in a graph, a submit is
worth 0.145-0.25 ms. The saving per submit depends on what else is in flight, so a figure from
one configuration does not scale to another.

The finder applies the program bound itself, in place of a build-and-shorten loop in the caller.
A layer is one or several programs, so the longest legal chained run is not a function of the layer
count. A chained run that is too long is split rather than refused. The bound rests on two queries,
`rocket_conv2d_int8_programs_rk3576()` and `rocket_conv2d_int8_chain_max_programs_rk3576()`.

`librocketnpu` contains no graph-execution object. The frontends do the scheduling. The library
decides which streams the hardware will run, which is the one thing a frontend cannot.

### Per-axis weights in a chained graph

In a chained graph, the per-axis requant's error does not compound. MobileNetV1-224 runs chained
with every convolution and depthwise convolution on the per-axis entry. Every one of its 29
compute layers is within one count of an exact per-axis requant. On each layer, 87.7% to 99.9% of
elements are identical to it. The chained network returns TFLite's top-1 and its whole top-5, and
no layer is refused [HW sweep, H96 MAX M9].

**Conditions:** H96 MAX M9. MobileNetV1-224 with its weights re-quantized per output channel, as
a TFLite int8 converter does, and its activations at the model's per-tensor scales and zero
points. The per-channel scale spread is 150x at the stem, 145x and 129x at the early depthwise
layers, and 105x at layer 25.

The single-operator worst case of 26.6 counts belongs to a synthetic 100x spread at ic=128, with
the whole layer in one tile. A real model's spread does not reach it. The planner sizes `C`
against the actual accumulator bound, and this model's wide-scale channels are also its
small-accumulator ones. On this model the planner never splits the output channels into tiles.
Its predicted
worst-case gain error is 0.003%-0.82% per layer, under the 1% target, so it keeps each layer in
one task and adds no submit.

> [!NOTE]
> A test that de-tiles every intermediate cannot detect a channel-order defect. The
> de-tile walks the sort permutation, so an output written in scale-sorted channel order reads
> back in model order and scores exact. The order is observable only where an output is consumed
> as a cube. A defect there shows in a whole network's label rather than in any layer's numbers.
> The same holds for any accuracy test whose reference is fed the tensor the hardware was.

**Not characterized:** a per-axis output consumed as a cube. Reaching it takes a real per-axis
model through a frontend.

## MobileNetV2

MobileNetV2-224 runs end to end at int8 on the RK3576. The graph has 65 layers:

- 36 convolutions
- 17 depthwise convolutions
- 10 residual adds
- A PPU pooling layer
- A classifier

Every one of the 64 compute layers is bit-exact against a CPU model of the RK3576's
arithmetic [HW sweep, H96 MAX M9]. The graph returns TFLite's top-1.

MobileNetV2 needs nothing new from the datapath. The add uses the lowering above, and the
graph needs two pieces of bookkeeping. One is a layer that names the layers its operands come
from, rather than assuming "the one before". The other is a buffer that outlives the
ping-pong, for the ten tensors a skip reads three to five layers later.

Against TFLite, the graph differs on 0.0573% of elements (3951 of 6896777), every one of them
by a single count. The rate is lower than MobileNetV1's 0.095%, over a graph twice as deep.
The ten adds are inside that one count, although the add does not use TFLite's arithmetic.
TFLite scales both operands onto a common intermediate. This lowering holds their scale ratio
in a pair of int8 weights and requantizes once.

The scale ratio is the term that can cost accuracy. On these ten skips, the best rational
approximation under 128, found by searching every denominator, is 1e-5 to 5e-4 relative.
Fixing the larger weight at 127 instead gives 1e-3 to 4e-3. At 1e-4, the second operand's
full int8 range moves an output by a tenth of a count. No add therefore exceeds the single
count that a requant difference already accounts for.

Fed the graph's own previous output at each layer, the disagreements compound as they do on
MobileNetV1, and the top-1 still matches. The tail of the top-5 reorders on this image.
TFLite's second through fifth classes all sit at probability 2/255, so their order is a
tie-break between equal values.

### Chain fragmentation on MobileNetV2

With every producer writing its own output buffer, the residual topology fragments
MobileNetV2's cube chain more than channel alignment does. MobileNetV2 then joins 32 of its 63
adjacent pairs into 16 chained runs of two or three layers. MobileNetV1 joins 25 of its 28 into 2 chained runs
covering 26 layers. The refusals, by reason:

| refused | why |
|---:|---|
| 10 | the consumer is an ADD, whose input is a host-built concatenation |
| 10 | the producer feeds a skip and so must leave a row-major tensor |
| 6 | the consumer's input channel count is not a multiple of 32 |
| 4 | the producer's output is not a feature cube (padded stride, or tiled) |
| 1 | the consumer is not a resident convolution |

Twenty of the thirty-one refusals come from the skip topology and six from the alignment, the
opposite of what the channel counts suggest. MobileNetV2's widths 16, 24 and 144 are not
multiples of 32, and they do cost joins. Three times as many are lost to the skips. An add
reads a concatenation the host builds, and the tensor that add reads has to survive in
row-major form until the add runs. An SoC with no alignment rule at all would still fragment
this graph into chained runs of three.

Both skip classes, the twenty refusals, close with one change. A producer writes its output at
a channel-group offset inside a larger buffer, as the next section describes. The other two
changes are the padding-channel rule and the producer's own group stride. With all three, the
same graph joins 62 of 63 pairs into one chained run covering 62 layers. The 63rd pair, the pool's
output into the classifier, joins once a pool can leave its output as a cube ("Pooling layers
as cube producers" below).

### Concatenation at a cube's base address

A convolution's feature base and its output base are both plain addresses on the RK3576
[HW sweep]. A program whose base moves by G channel groups inside a bigger buffer has these
properties:

- It produces the same output as at offset 0.
- It writes only its own slice.
- It leaves every byte outside that slice untouched.
- It composes with the row-task offset already added to the same register.

The two skip refusal classes therefore close in software. Allocate one buffer per add, and
give operand A the low slice and operand B the high one. Point the two producers at those
slices with `rocket_conv2d_int8_cube_out_at_rk3576()`. The add then reads the whole buffer as
its feature cube. No concatenation is built, and the skip source does not have to materialize
a row-major tensor for it. The skip source's output is the second operand, in place, and the
layer after it reads the same slice back as its own cube.

`librocketnpu` supports this through buffer ownership rather than through the encoding. A
handle can be told to write into a caller's buffer instead of allocating its own. The
functions `rocket_rk3576_cube_alloc()` and `_cube_slice()` provide that buffer and its views.
A `rocket_rk3576_cube` contains a byte offset that `_cube_in_` and the pooling entry honor on
the read side.

The PPU is a separate program with its own DMA. A pool over a producer's high slice is
bit-identical to the same pool over that producer's row-major output.

The scheme has three requirements beyond the address. The first is a hardware bound, and the
other two are bookkeeping:

- **A slice starts every sixteen channels.** Sixteen channels share every atom, so channel 24
  is not an address. A concatenation of two 24-channel operands therefore puts operand B at
  channel 32, with zero weights in the gap. Zero weights contribute nothing at any content, so
  the same weight matrix, descriptor and CPU reference serve both layouts.
- **The wiring is a pair, refused as one.** Operand B's producer is also the layer before the
  next block's expand convolution. Placing it therefore means that layer has to read the same
  slice as its cube. A producer that leaves no row-major tensor for a consumer that cannot take
  a cube would leave the graph with nothing at all.
- **A BO is named once per job.** The `rocket` driver locks each BO through `drm_exec`.
  Locking one twice fails with `EALREADY`, which rejects the submit. A layer reading one slice
  of a concatenation buffer while writing another is exactly one BO in both roles. That BO is
  named as the output, and a chained stream deduplicates its whole BO list the same way.

> [!CAUTION]
> `librocketnpu` stamps each output with the 0xA5 sentinel, and re-stamps it at the end of the
> call that read it. A cross-layer kick likewise re-stamps every interior layer's output inside
> its verify bracket. Both are safe only where the next kick rewrites the output before anything
> reads it. A concatenation slice, or an output with a distant second reader, is read later by a
> program outside the chained run that wrote it. A re-stamp replaces that output with 0xA5, and the
> reader computes an output of the correct size from it, unreported until the next materialized
> layer. Software must stamp such an output at the start of its producer's next call instead.

Software must take a placed producer's slice from the producer's own cube, not from the slice
the join pass built. The two name the same bytes, but only the producer records what its tail
holds. The consumer that reads operand B's slice back can have a channel count that is not a
multiple of 32. Taken from `rocket_conv2d_int8_cube_of_rk3576()` after the producer is placed,
the slice wires all ten of MobileNetV2's adds. Taken from the join pass, it wires nine.

#### Channel alignment and the weight zero point

The DPU's coefficient-group B term adds `B*sum(x)` over every tap of the whole programmed
group, padding channels included. A join at a channel count that is not a multiple of 32
therefore depends on the weight zero point, not on the channel count. A handle whose `ic` is
not a multiple of 32 fills the channels past `ic` with its border constant. A producer's output
instead holds whatever the DPU wrote into its own padded output channels. Those channels'
weights are the zeros the cube was memset with, so their MAC contribution is zero whatever
they hold. At a non-zero weight zero point, the B term is the one contribution from those
channels that is not zero.

A symmetric weight quantization therefore joins at any channel count, and every per-axis
TFLite filter is symmetric by construction. A per-tensor filter can have a zero point, as every
per-tensor depthwise filter does ([../chips/rk3576.md](../chips/rk3576.md) §"The depthwise
weight zero point"). An asymmetric one from
a direct producer also joins, because that producer's own partial output group is at its output
zero point. The consumer's fold assumes that constant (see "What a producer leaves past its live
output channels" above). An asymmetric one from a depthwise producer joins too, once its output
holds the deeper tail the host fills at pack time.

No channel count refuses a join on the RK3576, and neither does a producer's padded group
stride. With a pool's output also a cube, what remains, on 63 adjacent pairs:

| refused | why |
|---:|---|
| 0 | every adjacent pair joins |

### Inference time by optimization

**Conditions:** H96 MAX M9, 100 inferences per measurement, warm, one hardware kick per chained run of
cube-joined layers.

| Configuration | ms/inference | submits |
|---|---:|---:|
| transient weights | 112.9 | 64 |
| resident weights | 26.1 | 64 |
| + tensors kept in cube layout | 18.2-18.5 | 64 |
| + cross-layer kick | 15.7-16.3 | 40 |
| + concatenation buffers | 13.8 | 37 |
| + the padding-channel join | 10.0-10.4 | 9 |
| + one bracket for the sentinel | 8.8-8.9 | 9 |
| + the depthwise producer's tail | 7.5-7.7 | 7 |
| + the depthwise producer's group stride | 6.9-7.1 | 3 |
| + a pooling layer inside the stream | 6.7 | 1 |

Every row produces byte-identical output, with the same logits, the same top-5 and the same 77
tasks. The time the concatenation buffers, the two padding-channel rows and the group stride
save does not come from the memcpys they remove. With them, an add, an unaligned channel count
and a 7×7 depthwise each sit inside a chained run rather than between two. The submit count
falls with each.

With every optimization, the whole graph is a single chained run. The pool runs inside the kick, with
the classifier behind it, which saves 0.1 ms here. The submit count falls and wall-clock time
does not, because at this point a marginal submit costs about 0.05-0.10 ms on this graph.

The same sequence on MobileNetV1 ends at 4.9-5.1 ms in 1 submit. The concatenation and
padding-channel optimizations do not change MobileNetV1 at all, because its widths are all
multiples of 32 and it has no skips. The sentinel bracket does reduce it (6.4 -> 5.4). The
group stride reduces it too, closing its two 7×7 depthwise joins (5.4 -> 5.1, 5 submits -> 3).
Resident weights remain the largest single optimization (4.3x), and no single layer dominates.

Per-output-channel weight scales also run on MobileNetV2, with the same result as on
MobileNetV1. Every one of the 53 weighted layers is within one count of an exact per-axis
requant, at real scale spreads to 200x. No layer uses output-channel tiling
[HW sweep, H96 MAX M9, 2026-07-31].

## ResNet-18

ResNet-18-224 runs at int8 on the RK3576. The graph has 32 layers:

- 21 convolutions
- 8 residual adds
- A max pool
- An average pool
- A classifier

ResNet-18 exercises the RK3576 differently from the two MobileNets. It is 3×3-heavy where both
of those are 1×1-heavy, and it also has:

- A 7×7 stem
- A max pool
- A fused ReLU on every add
- Explicit torchvision padding

The padding puts a lead pad of 3 against a trailing 2 on the stem, an asymmetry in the
direction TFLite's SAME never produces.

No per-tensor quantized ResNet-18 is published, so the model is built from the timm
`resnet18.a1_in1k` checkpoint and quantized per tensor by TFLite post-training quantization.
Its tail is `AveragePooling2D 7x7 -> a 1x1 CONV_2D classifier -> softmax`, which is
MobileNetV1's tail and keeps `FULLY_CONNECTED` and `MEAN` out of the op set. The model deviates
from torchvision in one place. The explicit 3-pixel border of conv1 is padded in the raw pixel
domain, so it is black where torchvision's is the channel mean. It is a 3-pixel frame on a
224-pixel plane, and every result is scored against this model's TFLite output.

Against TFLite, the graph differs on 0.0386% of elements (1327 of 3438568), all by a single
count [HW sweep, H96 MAX M9]. The rate is the lowest of the three networks. Every add, every
3×3 convolution, both pools and the classifier are bit-exact against a CPU model of the
RK3576's arithmetic. The chained graph returns TFLite's top-1.

| Configuration | ms/inference | submits |
|---|---:|---:|
| transient weights | 144.9 | 31 |
| resident weights | 35.4 | 31 |
| + tensors kept in cube layout | 32.8 | 31 |
| + cross-layer kick | 18.3 | 25 |

**Conditions:** H96 MAX M9, NPU 786 MHz, 2026-08-02, the stem on the direct lowering, the write
guard bracketed per layer.

Every row produces byte-identical output, with the same logits, the same top-5 and the same 38
tasks. At the cross-layer kick, with the join refusals open, the chain fragments far more than
either MobileNet's. It makes 18 joins of 30 adjacent pairs, in 5 chained runs covering 11 layers,
against MobileNetV2's 62 of 63. Every width here is a multiple of 64, so the channel counts
cause none of the refusals. Each refusal class closes in software, as "The three refusal
classes" below describes.

With every optimization in "Per-optimization decomposition" below, ResNet-18 runs as one kick
in one submit at 9.2 ms. With the packed-image stem and one guard bracket per BO, the same
graph runs in 6.6 ms at the same clock (2026-08-03, "Composition of a graph's wall-clock time").

### Per-optimization decomposition

**Conditions:** H96 MAX M9, NPU 786 MHz, 100 inferences per measurement, spread between
repeats under 0.3 ms, 2026-08-02. The stem is on the direct lowering and the write guard is
bracketed per layer, as in the table above.

The join count and the submit count do not predict the time an optimization saves. Four
library options switch independently, so the steps above decompose:

| Configuration | joins | submits | ms |
|---|---:|---:|---:|
| no cube layout | 0 | 31 | 34.5-34.6 |
| + adjacent-pair joins | 11 | 31 | 22.8-23.1 |
| + cross-layer kick | 11 | 26 | 20.7-20.8 |
| + concatenation-wired adds | 24 + 8 far | 17 | 13.7-13.9 |
| + shared outputs | 27 + 11 far | 11 | **10.6** |
| + the chained-run rule, not the pair rule | 27 + 11 far | 5 | 9.9 |
| + a pooling layer inside the stream | 27 + 11 far | **1** | **9.2** |

Eleven adjacent-pair joins save about 11.6 ms at a constant submit count. All of that is host
transposes and the buffer machinery around them, and none of it is dispatch. The kick's five
submits there save 2.2 ms. The transposes are the larger term by about 5x, so the submit count
alone does not predict the saving. A submit costs 0.15-0.45 ms depending on which comparison
measures it, and a join 0.10-6.80 ms depending on the bytes it moves.

The last two configurations change dispatch alone, with no join change, no task change and the
same 38 programs. They take the graph from 11 submits to 1 for 1.4 ms, about 0.14 ms per
submit [host-computed].

> [!NOTE]
> A marginal submit costs less as a graph runs out of submits. Measure it on the graph at hand,
> rather than scaling a figure taken on a graph of thirty submits.

> [!NOTE]
> Compare configurations at one iteration count, of more than 10 inferences. The configuration
> with no cube layout measures 43.3-45.1 ms over 10 inferences and 34.5-34.6 ms over 100,
> because its first few inferences cost more than the rest. Every other configuration in the
> table measures the same at both counts. A comparison across counts overstates that one
> optimization by 1.8x.

A join removes two host transposes over the producer's whole output, so a refused join's cost
scales with its bytes. A 56×56 pair moves 64x what a 7×7 one does. The twelve refusals at the
18-join stage of the ResNet-18 table above, by bytes:

| refused | KiB | why |
|---:|---:|---|
| 6 | 514 | the producer feeds a skip and must leave a row-major tensor |
| 4 | 368 | the consumer is an ADD reading a host-built concatenation |
| 2 | 196 | the consumer is not a resident convolution |

The 18 joins present at that stage move 2278 KiB. The byte ratio orders the refusal classes and
does not bound the saving, because the cost per join is not uniform. The two classes already
closed differ by 8x. The cost of one join is measured as a finite difference. The option
`ROCKET_RK3576_NET_NOJOIN=<producer>` refuses exactly the join that producer makes, and leaves
every other one intact.

A join's cost is its bytes, across a 32x range. Each row below removes one of the 18 existing
joins, against a base of 18.40 ms:

| producer's tensor | joins removed | ms |
|---:|---:|---:|
| 784 KiB (the stem) | 1 | 6.80 |
| 196 KiB | 1 | 0.57 - 1.10 |
| 98 KiB | 1 | 0.53 - 0.63 |
| 49 KiB | 1 | 0.20 - 0.33 |
| 24.5 KiB | 1 | 0.10 - 0.17 |

**Conditions:** H96 MAX M9, 100 inferences per measurement, spread under 0.2 ms.

The 18 single-join removals sum to 15.0 ms over 2278 KiB, about 6.6 us/KiB. With the kick off
and submits fixed at 31, the difference is all host time. The removals then sum to 17.25 ms
over the same bytes [HW sweep, H96 MAX M9]. A join's cost is therefore the bytes it moves, and
the stem's join saves more than any other in the graph.

In the per-layer profile of the 25-submit configuration, 18.4 ms over 38 tasks, the largest
layer is the 7×7 stem at 2.94 ms (19.4%). The next nine take 0.53-1.02 ms each. Every
convolution is one output-channel tile and one task, so no layer's time goes to `oc` tiling.
The two MobileNets give the same result.

### The three refusal classes

All twelve refusals close in software, and none of them needs a register change. Closing them
takes ResNet-18 from 25 submits to 11 on the same 38 tasks. The output is byte-identical, with
the same logits, the same top-5 and the same TFLite distance. Each class needs a different
change.

#### Skip sources with a distant reader

When every layer that reads a skip source can read a cube, the skip source can write one. The
requirement that a producer read by a later layer materializes a row-major tensor is a
host-buffer rule. It holds only where that later reader, three layers on, needs a row-major
tensor.

A resident handle has its own output buffer, and nothing writes there until that handle runs
again. That output therefore already outlives the whole inference, which is exactly the
lifetime a distant reader needs. Two readers of one output read the same cube, described twice,
with no second copy.

ResNet-18's identity blocks end in an add whose output is read twice. The next block's 3×3
convolution reads it, and so does that block's 1×1 downsample three layers later. Both are
ordinary convolutions. There are three such producers, 343 KiB.

A shared output has the sentinel hazard in the Caution under "Concatenation at a cube's base
address" above. The declaration `rocket_conv2d_int8_cube_shared_rk3576()` moves the stamp to the
start of the producer's next call, where cube-out already puts it. A handle writing into a
caller's buffer is already that case and needs no declaration.

#### Concatenation buffer placement

A concatenation buffer's placement rule cannot assume that either operand comes from the layer
before the add. Keying the buffer on operand A's producer being the layer immediately before
the add fits MobileNetV2's shape. It is the exact reverse of a ResNet block that changes width.
There, operand B is the 1×1 downsample immediately before the add, and operand A is the 3×3 leg
past it.

Each add gets a dedicated buffer, and each slice has exactly one writer, so the order the two
producers run in does not matter. The placement needs two conditions. Both producers precede
the add, which they do by construction, and every other reader of either producer can read the
slice back. The rule stated that way wires both families. ResNet-18 has 8 concatenation-wired
adds under it, against 4 under the operand-A keying, and MobileNetV2 has 10 under both.

#### Pooling layers as cube producers

A pooling layer is a cube producer. The PPU's surface stride is `round4(ow*oh)` rather than the
plane. The CNA's DDR channel-group jump is a register the regcmd encoder fills, and the RK3576 honors
it at any value at or above the plane. The cube descriptor therefore contains the pool's
stride, and a convolution consumer joins a pool at any plane.

The PPU's destination channel-group stride is `0x607C`, honored at any value at or above the
plane ([../chips/rk3576.md](../chips/rk3576.md) §"PPU destination and source addressing").
`rocket_pool_int8_cube_out_at_rk3576()` takes the stride from the slice and refuses one below
the plane.

The last two refusals are both the max pool, which feeds the next convolution and a residual
add three layers on. Both close once the PPU can leave its output in place. The functions
`rocket_pool_int8_cube_out_rk3576()`, `_cube_of_` and `_cube_out_at_` are the pool side of the
join.

#### Joins and chained runs

With those three changes, every adjacent pair of all three networks joins. ResNet-18 joins 27
of its 30 index-adjacent pairs, MobileNetV1 28 of its 28 and MobileNetV2 63 of its 63.
ResNet-18's other three index-adjacent pairs are not dataflow edges, because its downsample
reads across them. The two MobileNets' inference times do not change (5.0 and 6.9 ms), so
their last unjoined pair, the pool's output into the classifier, saves nothing.

Closing every join does not make a graph one chained run. With all 27 joins in place, ResNet-18 runs in
11 submits under a chained-run finder with two rules the hardware does not impose. A finder that grows
a chained run only while consecutive pairs are joined strands a shortcut convolution reading three
layers back. A finder that treats a pooling layer as a break splits the stream, and a pool need
not be a break.

The sections "The chained-run finder" and "A pooling layer inside the stream"
above replace both rules. With them, the graph is one kick in one submit. A pair can be joined
and still sit at a chained-run boundary, so the chained runs need counting as well as the joins.

The cost of each class, measured with its option off and the other two on:

| Option off | joins | submits | ms |
|---|---:|---:|---:|
| shared outputs (`ROCKET_RK3576_NET_MULTI=0`) | 24 + 8 far | 17 | 13.7-13.9 |
| concatenation buffers (`ROCKET_RK3576_NET_CAT=0`) | 14 + 3 far | 26 | 18.9-19.0 |
| cross-layer kick (`ROCKET_RK3576_NET_KICK=0`) | 27 + 11 far | 31 | 13.4-13.8 |
| none (everything on) | 27 + 11 far | 11 | **10.6** |

**Conditions:** H96 MAX M9, 100 inferences per measurement, 2026-08-02.

Summing each refused pair's cost at its own size gives about 4.4 ms of 18.4. The measured
saving is 7.8 ms, 18.4 -> 10.6, 1.8x the summed 4.4 ms [HW sweep, H96 MAX M9, 2026-08-02]. Two terms
outside the refused-pair table make up the difference. The closures also remove 11
non-adjacent joins moving 1078 KiB, a distant reader's scatter, which no adjacent-pair row
contains. They also merge chained runs, taking 25 submits to 11. The refused-pair table therefore
orders the classes and does not bound the saving [HW sweep, H96 MAX M9, 2026-08-02].

### Operand producers in a graph

A graph layer's operands need not come from the layer before. A feed-forward graph's operand
A is always the previous layer's output, and a residual graph's is not. MobileNetV2 reads a
skip three to five layers back on operand B, and the main line from the layer before. A
ResNet block that changes width reads them the other way round. Operand B is the 1×1
downsample immediately before the add, and operand A is the 3×3 leg past it. The downsample
reads the block input again three layers later, which is a distant read on a convolution's
first operand.

> [!CAUTION]
> A graph runner can thread operand A from the previous layer and give a private buffer only to
> an add's second operand (`src2`). Such a runner feeds three of ResNet-18's layers the wrong
> tensor. Each layer produces a full, correctly sized, plausible output, nothing faults, and
> the graph returns a confident label for an unrelated class. A per-layer test that feeds each
> layer TFLite's reference input scores every layer exact. Software must give a buffer of its
> own to every producer read by anything other than the next layer. Software must resolve both
> operands of every layer by its producer, on every branch.

Whether a graph exposes this can be counted before any run. The count is the number of layers
that take operand A from a producer other than the layer before. It is 0 for MobileNetV1 and 0
for MobileNetV2, against 3 convolutions and 3 other layers for ResNet-18. Where the count is
zero, resolving by producer and by position are the same function, so a wiring defect passes
every test on that graph.

### Per-axis requant on ResNet-18

ResNet-18 re-quantized per output channel returns TFLite's top-1 on the RK3576, as both
MobileNets do. All 31 compute layers are within one count of an exact per-output-channel
requant, at spreads to 42x, and only the tail of the top-5 reorders. An unrelated class at
the output indicates the operand wiring above, not the requant.

**Conditions:** one input image.

A per-layer comparison cannot separate the two causes. A layer fed the wrong tensor is
scored against a reference built from that same wrong tensor, so it reports exact. The graph run
on the host with only the operand resolution varied separates them. With operands resolved
by producer, the per-axis graph puts TFLite's top-1 2 counts behind its own top-1. With
operand A threaded, TFLite's top-1 falls 120 counts behind.

The 2 counts that remain are a near-tie, and the network's own margin is the scale to read
them against. Per tensor, this image puts `suit` 7 int8 logit counts
ahead of `bulletproof vest`, and the exact per-axis chain puts them 2 counts the other way.
The RK3576's per-axis path, whose fixed-point requant is itself up to one count per layer
from the exact one, returns `suit`. Which side a per-axis ResNet-18 returns is inside the
resolution of the requant. A rank swap inside the top-5 at such a margin is the requant's
accuracy, and a class outside the top-5 indicates a different graph.

A per-axis chain of this depth does not compound its per-layer count into a class change on
this image.

**Not characterized:** other images, other network families, and a graph whose top two
classes are closer than the 7 counts this one has to spare. On such an image the swap is
expected and is not a defect. The result also does not exercise the planner's tile option,
which the planner did not take here.

## Inception V1 and the concatenation topology

Inception V1 (GoogLeNet) at 224, int8, runs on the RK3576. The graph has 82 layers: 58
convolutions, 13 max pools, a 7×7 average pool, 9 concatenations and a classifier. It
differs from the three earlier networks on these axes:

- A concat/split topology: nine four-operand concatenations along the channel axis.
- Nine producers with four readers, which is exactly the maximum the join passes accept.
- A k=7 average pool, where the earlier pooling tests use 2, 3 and 5.
- Thirteen pooling layers, against one or two in each of the other three networks.
- 18 convolutions and 18 other layers whose operand A comes from a producer that is not the
  layer before. ResNet-18 has 3 and 3, and both MobileNets have 0 and 0.
- Channel counts of 480, 528, 832 and 1024, one of which (528) is not a multiple of 32.

The model arrives already quantized per tensor, from the same `google-coral/test_data` as
the MobileNets.

All 81 compute layers are bit-exact against a CPU model of the RK3576's arithmetic. The graph
returns TFLite's top-1, and differs from TFLite on 721 of 5881241 elements (0.0123%), each by
one count. The encoding, the padding lowering and the requant needed no change for this
graph.

A concatenation whose operands are already in its own quantization is placement, not
arithmetic, and all nine here qualify. Every operand tensor has the output's scale and zero
point. The concatenation is then a copy on the host and no work on the NPU, like a residual
add's operand pair. A placed cube slice must start on a 16-channel atom, and all nine
operands do.

The topology fragments the chain more than any other graph measured on the RK3576, and none
of the fragmentation comes from the hardware. With the concatenations built on the host, 34
of 80 adjacent pairs join. MobileNetV1 joins all 28 of its pairs, MobileNetV2 all 63 and
ResNet-18 27 of 30.
The graph then runs in 69 submits with a single 4-layer kick, at 19.8 ms. The same graph with
no resident weights and no cube layout takes 159.7 ms.

Thirty-six of the graph's 82 layers are skip sources, where ResNet-18 has three. A skip
source here is a branch output read by a concatenation four layers later. With the
concatenations wired as placed slices (below), every one of the refused joins closes, and the
graph runs in 10.8 ms in one kick.

### Pooling layers and the host transpose

A pooling layer's host cost is its cube transpose. With the scalar host scatter, the six most
expensive layers after the stem's kick are all max pools. They cost 2.76 ms at `480x28x28`,
1.88 at `256x28x28`, then 1.21, 0.98, 0.75 and 0.72. MobileNetV1's whole pooling bucket is
0.17 ms. The table splits the cost of all thirteen pooling layers of one inference, in
microseconds, as `ROCKET_RK3576_POOL_PROF=1` reports it:

| bucket | scalar loop | NEON block | what it is |
|---|---:|---:|---|
| the host cube scatter | 8556 | 765 | row-major CHW into NC1HWC2 |
| the completion wait + output cache maintenance | 1856 | 1860 | `PREP_BO` on the output, then the sentinel scan |
| the sentinel stamp | 454 | 467 | |
| the de-tile | 180 | 63 | |
| the submit ioctl | 160 | 160 | |
| **total** | **11206** | **3316** | |

The submit is 160 us across all thirteen layers, 1.4%, so no pooling layer here is
dispatch-bound. A pool that costs ten times a convolution of the same shape is paying for its
transpose, not for its program.

The scalar scatter is a per-element loop with a strided sixteen-byte store, at about 145
MB/s. The convolution path's 16×16 NEON block transpose runs in GB/s. Through that block, the
same 1.18 MiB takes 765 us, about 1.6 GB/s. The block is in `rocket_rk3576_cube_pack.c`, and
both entries call it.

With the NEON block, the graph goes from 27.2-27.6 to 20.0-20.1 ms, 1.36x, on the same 69
submits and 77 tasks. The output is byte-identical per layer against the CPU model. The same
69-submit graph measures 19.8-19.9 ms as the control of the comparisons under "Concatenation
placement", later on the same day, 2026-08-02.

The other three graphs do not move. Their pools are cube-joined at both ends or are a single
`1024x7x7` tail, so none of them has a transpose. The cost is a property of the join state,
not of the pooling program. The four cube-in pools here already cost 0.12-0.39 ms, where the
nine un-joined ones cost 0.13-0.51.

On the pack side, a full channel group takes the block and a partial one keeps the scalar
loop. The split follows the padding contract and is not a tail optimization. The block
stores whole sixteen-byte atoms. On a partial group it would fill the cube's dead lanes with
a value that the PPU then reduces into output channels past `c`. The pooling entry leaves
what a consumer finds there undefined. Every pooling layer of the four networks here has a
multiple of sixteen channels, so the block covers all of them.

The convolution side's own buckets, summed over the 55 per-call profiles of one inference,
are these [HW sweep, H96 MAX M9, 2026-08-02]:

- Submit-and-wait: 7.37 ms.
- Readback: 1.57 ms.
- Feature scatter: 1.37 ms.
- `plan+gen`: 0.43 ms.

### Per-layer scoring overhead

Scoring every materialized layer against its reference tensor on every iteration adds a full
per-element pass over each tensor. The pass falls outside the per-layer timer and inside the
wall-clock time. On
Inception V1 that pass is 10.1 ms of a 28.9 ms result, 35%. On a graph that runs as one kick
with one materialized layer it costs nothing measurable, so the overhead depends on the
graph. The Inception V1 timings here score the first iteration only.

### Concatenation placement

Four properties decide how a concatenation is placed:

- A cube's base is a plain address on both sides of a convolution.
- A handle writes its register channel count, rounded up to 32.
- The PPU reduces within a channel and writes only the channels it has.
- The RK3576 honors write-after-write between two programs of one stream [HW sweep, H96 MAX
  M9, 2026-08-02], and read-after-write between two chained programs.

With the concatenations built on the host, Inception V1 refuses 46 joins. Forty-five of them
come from the concatenation topology, in two shapes. The first is 27 branch outputs that have
to materialize, because a concatenation several layers later copies them on the host. The
second is 18 pairs whose producer or consumer is the concatenation itself. A host
concatenation is not a resident convolution, so it cannot be either side of a join. Neither
shape comes from the hardware, and both close the same way the residual adds do.

The 46th is a channel-alignment refusal: the 528-channel max pool below, whose consumer has a
non-zero weight zero point. A round-32 destination for the pool closes it, as below.

Each concatenation gets one buffer, with a slice per operand. The four producers write their
own slices of one allocation with `rocket_conv2d_int8_cube_out_at_rk3576()`, and every consumer
reads the buffer as its feature cube. The concatenation then costs nothing: no copy, no
transpose and no program.

A chained run also spans a placed concatenation. A layer with no program would otherwise break every
chained run it sits in, and on this graph that is every chained run there is. The library therefore has a
third node kind, `rocket_chain_node_rk3576.placement`: a layer the caller states has no
program and no host work. A placement node has these properties:

- It contributes zero programs.
- It is exempt from both chained-run conditions, since it neither reads a cube nor leaves one.
- It must be interior to a chained run.
- It is dropped from the chain object entirely, so the layout, the stamp, the verify and the redo
  never see it.

The function `r76_run_fed` then matches the consumer against the producer of operand zero,
whose output base is the buffer base.

Placement has four requirements, and three of them are not in the residual add's version.

First, the producers overlap. An operand whose count is not a multiple of 32 writes past its
own live channels into the head of the next operand's slice. `cube_out_at` refuses a slice that
is smaller than what its producer writes.

> [!CAUTION]
> An operand whose channel count is not a multiple of 32 writes into the head of the next
> operand's slice. Software must size each slice for what its producer writes. Software must
> run the next operand's program after it, so that its program overwrites those bytes.

Write-after-write holds on Inception V1: two of the nine concatenations reach it, at 16
channels each, and the graph is byte-identical with them wired. Wiring those two is worth
5.0 ms (15.9 -> 10.9), a third of the placement gain [HW sweep, H96 MAX M9, 2026-08-02].

> [!NOTE]
> The overlap weakens the library's write guard for those producers. The guard asks whether
> any byte of a task's extent differs from the sentinel. The extent covers the padding
> groups the next producer overwrites, so a program that stopped mid-stream can pass as one
> that wrote. The guard's per-kick form already accepts that blind spot. A whole submit
> corrupted by an earlier wide output is still detected, since every other layer of the kick
> stays unwritten too.

Second, the last producer's overhang makes the tail sound. Its padding channels hold its
own output zero point, which is the concatenation's zero point and so every consumer's input
zero point. A consumer whose channel count is not a multiple of 32 can therefore be told what
its tail holds, instead of having the join refused. The groups no producer reaches are
filled with that constant once at join time, bracketed.

Third, a slice is sized for a writer and a view for a reader. A producer's slice covers the
groups it writes. A consumer's view is the whole buffer narrowed to the concatenation's
channel count, because a feature DMA walks `round32(ic)` and the buffer has those groups. The
narrowing is `rocket_rk3576_cube_declare_tail()`. By contrast, `cube_slice` sizes a slice from
its live channels, which is right for a writer and one group short for a reader.

Fourth, the whole concatenation is one decision. A producer that placed its output for a
consumer that then refused the buffer would leave no row-major tensor at all. The whole
concatenation is therefore wired or none of it is, by the residual add's pair rule over up to
eight relations instead of three.

A pool's output is a group short, and what lies past it is whatever the input cube's padding
held. On Inception V1 one such refusal costs four kicks. A 528-channel max pool feeds a
convolution with a non-zero weight zero point, and the pool cannot declare its tail. With the
nine concatenations wired, the graph still runs in 16 submits while that refusal stands.

A chained run cannot span a layer that leaves a row-major tensor, so that one pair splits the graph
at layer 59. Every later module's concatenation takes its first operand from a layer before
the split. The chained-run rule needs that operand inside the same chained run. Giving the pool a
round-32 destination whose tail holds its output zero point closes the refusal. The
destination is the concatenation buffer with one operand. It takes the graph from 16 submits
to 1, and it is worth 1.6 ms, although the join itself moves 101 KiB.

Alone, the tail fix is worth nothing (19.8 against a 19.9 ms control). The two changes are
superadditive, because the tail fix helps only once the concatenation wiring has removed
everything else in the way.

The whole graph is then one hardware kick in one submit. Every adjacent pair that is a
dataflow edge joins: 53 of 80 index-adjacent pairs, and the other 27 are not edges. Another
54 non-adjacent joins move 3776 KiB, and the graph is one chained run of 81 layers. Wall-clock time
goes 19.8 -> 10.8-10.9 ms and the submits 69 -> 1, on the same 77 tasks. The logits, the
top-5 and every materialized layer are byte-identical. The three older graphs do not move, and
each stays one submit [HW sweep, H96 MAX M9, `bench 100`, 2026-08-02].

The 9.0 ms gain splits into the terms below, including one cost that the change adds. Both
ends are profiled with `ROCKET_RK3576_INT8_PROF=1` and `ROCKET_RK3576_POOL_PROF=1`, per
inference:

| term | host concatenations | placed | delta |
|---|---:|---:|---:|
| host cube transposes (feature scatter + de-tile, conv and pool) | 5.28 | 0.60 | **-4.68** |
| submit and wait (55 conv calls, 13 pool calls, one 4-layer kick) | 12.59 | 5.59 | **-7.00** |
| `plan+gen` | 0.57 | 0 | **-0.57** |
| the write guard (`stamp` + `verify`) | 1.37 | 4.63 | **+3.26** |
| **total** | **19.82** | **10.82** | **-9.00** |

Placement increases the write guard's cost. A placed slice cannot be re-stamped inside the
verify bracket, because something outside the stream reads it later in the same inference.
Each of the 37 placed producers therefore takes a second `PREP_BO`/`FINI_BO` pair at the start
of the next call. A one-kick graph also verifies all 72 outputs, where the per-layer path
verifies them inside its own submit bucket.

The guard is 4.63 ms of the 10.8 (43%). With `ROCKET_RK3576_I32_SENTINEL=0`, the graph runs in
6.2 ms, which confirms the guard's share independently. With `ROCKET_RK3576_GUARD_PER_KICK=1`
it also runs in 6.2, the time with the guard off, as on all three older graphs.

## Inception V3 and test symmetry

Inception V3 at 299, int8, runs on the RK3576, and its non-square kernels and padded pools
vary axes that the earlier networks hold fixed. The graph has 131 layers: 95 convolutions, 4
max pools, 10 average pools, 15 concatenations, six requantization edges and a classifier.
All 130 compute layers are bit-exact against a CPU model of the RK3576's arithmetic, and the
graph returns TFLite's top-1. It differs from TFLite on 4966 of 15122569 elements (0.0328%),
each by one count, in line with the four graphs before it (0.0123% to 0.095%).

The graph runs in 33.4-33.7 ms at `bench 100` and NPU 786 MHz [HW sweep, H96 MAX M9,
2026-08-03]. The figure at 594 MHz is in [../chips/rk3576.md](../chips/rk3576.md) §"NPU clock
rate". The library's divisor-lag check scores the last channel group alone, which every recorded
occurrence of the lag reaches.

The graph runs in 48 submits, against one for each of the other four graphs. Its
concatenation topology and its four-operand branches leave far more of it outside a kick: 27
layers. The chained-run finder keeps out 9 of them, the padded average pools. "Inception V3 chained-run
fragmentation by reason" below counts the 27 by reason.

### The join ratio's denominator

The PPU's destination has no line stride, so a column slice cannot write into one plane.
That limit of the RK3576 is behind the four joins Inception V3 refuses, the column-split
pools' output side (1214 KiB).

A join ratio's denominator is dataflow adjacency, not index adjacency. The figure "n join(s)
of m adjacent pair(s)" is a ratio only if every pair in m is a relation a join can remove. On
a feed-forward chain, index adjacency and dataflow are the same relation. On a branchy graph,
most index-adjacent pairs are not edges at all. An Inception module's four branches sit
consecutively in the layer table, and none of them reads the one before it.

**Conditions:** H96 MAX M9, `bench 100`, measured 2026-08-03 [HW sweep]. The pairs counted
are those where layer j names layer i as an operand.

| network | joins | dataflow-adjacent pairs | refused | index-adjacent non-edges |
|---|---:|---:|---:|---:|
| MobileNetV1 | 28 | 28 | 0 | 0 |
| MobileNetV2 | 63 | 63 | 0 | 0 |
| ResNet-18 | 27 | 27 | 0 | 3 |
| Inception V1 | 53 | 53 | 0 | 27 |
| Inception V3 | 90 | 94 | **4** | 35 |

Four of the five graphs join every dataflow-adjacent pair, and only Inception V3 refuses any.

The two refusals that placement can reach are joined. Both are the sides of one 768×17×17
concatenation, the only one of the five networks' 39 that is not placement without this
change. Joining them saves 2.6-3.1 ms.

Most of that gain is outside the joined pair. The buffer also gains five non-adjacent readers
(65 -> 70 joins, 8249 -> 9035 KiB). One of them, the 768×17×17 average pool at layer 50, is
where most of the millisecond is: 2.46 -> 0.83 ms. It drops because a cube input also takes
its divisor-lag fire rate from 48.2% to 0-1% [HW sweep]. The drop matches the recorded
property that the lag's rate follows the submit structure rather than the geometry.

> [!CAUTION]
> A join pass that joins index-adjacent layers without a dataflow check joins a non-edge
> pair whose shapes agree. The consumer then reads a tensor that nothing in the graph says it
> reads, and nothing faults. Every other clause of the join pass is a geometry or an
> ownership question. Software must join a pair only where the consumer names the producer
> as an operand.

### Symmetric test axes

Four register fields and behaviors of the RK3576 cannot be checked along an axis that the
earlier networks and tests hold constant. Each has the same signature, a full, correctly sized,
entirely plausible output, and none raises a fault or a refusal. Each row names the axis, what
Inception V3 varies, and the behavior:

| axis the earlier tests hold constant | what Inception V3 varies | behavior |
|---|---|---|
| every convolution kernel square | 1×7, 7×1, 1×3, 3×1, 34 layers | the CNA's kernel word halves (width and height) |
| every average pool unpadded | nine 3×3 stride-1 SAME pools | `AVG_NOPAD` moves the divisor and not the sum |
| every pool plane <= 112 at stride 2 | 147 at stride 2, 35 at stride 1 | a per-task output-width allowance |
| every pool divisor constant over position | a per-position divisor, from that mode bit | the divisor lags one output window |

> [!NOTE]
> A test finds a transposed or mis-keyed field only along an axis that separates it from a
> plausible wrong value. For each packed register field and mode bit, name that axis, and
> check that some test point varies it.

The PPU's `0x6034` packs `(sy-1)<<20 | (sx-1)<<16 | (kh-1)<<8 | (kw-1)`, as labeled. Kernels
driven as rectangular pairs at one plane confirm it [HW sweep, H96 MAX M9]. A transpose
would map each test point of a pair onto the other.

The PPU's per-task output-width allowance is a function of the kernel height and the vertical
stride, which a square table cannot show (see [../chips/rk3576.md](../chips/rk3576.md) §"The
pooling output-width allowance"). An allowance keyed on the width fails every pool with `kh >
kw` and passes every pool with `kh < kw`. Keyed on the height, all eight rectangular test points
are bit-exact, and so are 21 other pooling test points [HW sweep, H96 MAX M9].

The other axes Inception V3 varies cost nothing, and all of these are exact:

- 8 VALID convolutions.
- Planes of 299, 149, 147, 73, 71, 35, 17 and 8.
- Three 5×5 kernels.
- Two 3-operand concatenations.
- 24 convolutions plus 26 other layers whose operand A is not the layer before, against
  ResNet-18's 3+3 and the MobileNets' 0+0.

### Requantization edges

A requantization edge is a TFLite `QUANTIZE` op where a tensor is consumed at a different scale
from the one its producer wrote. On Inception V3 it follows a concatenation whose operands were
calibrated somewhere the consumer is not. Inception V3 has six of them, all pure rescales at zero
point 0.

The op lowers onto a depthwise 1×1 identity convolution. With a unit weight at unit scale, the
convolution's own epilogue computes
`out = requant((in - in_zp) * in_scale / out_scale) + out_zp`, which is exactly TFLite's kernel.
The layer can write straight into a concatenation's slice like any other producer. A direct 1×1
convolution over an identity matrix computes the same function. Its cost is quadratic in the
channel count, where the depthwise form's is linear.

In `librocketnpu` a requantization needs no new layer kind, no new blob field and no new library
entry.

### Inception V3 chained-run fragmentation by reason

Four reasons keep a layer of Inception V3 outside any chained run. Only one is a property of
the RK3576: a padded average pool whose divisor can lag. `kick_build()` reports the reason per
layer. 27 layers sit outside any chained run, and each is a submit:

| count | why |
|---:|---|
| 9 | it is a padded average pool whose divisor can lag |
| 2 | it leaves no cube, so the layer after it would need a host de-tile |
| 3 | it reads no cube any earlier member of the chained run wrote |
| 13 | it is joined at both ends but is the only program between two breaks |

**Conditions:** H96 MAX M9, `rocket` 1.6.0, NPU 594 MHz, resident weights and the cube layout.
The graph runs as 16 chained runs covering 94 layers, at 48.4-49.0 submits an inference [HW sweep,
2026-10-12].

The other four graphs are each one chained run covering every layer. The remaining reasons come from
the test harness's lowering and from this graph's topology, as do ResNet-18's and Inception
V1's refusal classes. The 13 in the last row are consequential rather than causal: they are what
the other 14 cost.

## Composition of a graph's wall-clock time

Each of the four graphs runs as one chain call in one submit, so the profile accounts for its
whole wall-clock time.

**Conditions:** H96 MAX M9, NPU 786 MHz, `bench 100`, resident weights, cube layout,
cross-layer kicks, one write-guard bracket per BO, `ROCKET_RK3576_INT8_PROF=1`, 2026-08-03. The
stems of ResNet-18 and Inception V1 run on the packed-image encoding, ResNet-18's with its pad
columns materialized.

| | ms | submits | tasks | submit + wait | verify | stamp | feature scatter | de-tile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV1 | 5.0 | 1 | 40 | 2.36 (47%) | 2.04 (40%) | 0.03 | 0.61 (12%) | 0.01 |
| MobileNetV2 | 6.6 | 1 | 77 | 2.88 (43%) | 2.93 (44%) | 0.26 (4%) | 0.61 (9%) | 0.01 |
| ResNet-18 | 6.6 | 1 | 33 | 4.38 (66%) | 1.32 (20%) | 0.78 (12%) | 0.17 (3%) | 0.02 |
| Inception V1 | 7.1 | 1 | 72 | 4.08 (58%) | 1.71 (24%) | 1.12 (16%) | 0.09 (1%) | 0.01 |

The write guard is the `verify` and `stamp` columns together. The table has no `plan+gen`
column, because a chain generates its whole stream once, at construction. A graph that is one
kick therefore does no per-inference regcmd work at all. Inception V3 is not in the table
because it is not one chain call. It runs as 16 kicks plus per-layer convolution and pooling
calls, 47.6-48.1 submits an inference [HW sweep, H96 MAX M9, `bench 100`, 2026-08-03]. Its
accounting is under [../chips/rk3576-hazards.md](../chips/rk3576-hazards.md) §"The pooling
path's read interval".

These graphs are not host-bound. Their two largest terms are the submit and the write guard,
and neither is host arithmetic. The host operand packing is 1-12% of the wall-clock time: one
feature scatter per graph, at the stem, because every other layer reads a cube.

### The write guard

The write guard is the second-largest term on every graph, and the largest on MobileNetV2. It
is a `PREP_BO`/`FINI_BO` pair per output per call, and both halves sync the whole BO. Its cost
is therefore proportional to the BO's bytes, however few bytes the check reads.

`ROCKET_RK3576_GUARD_PER_KICK=1` checks one output per kick rather than every layer's. It
removes exactly the share of the guard that the graph keeps inside kicks. Its coverage
therefore depends on the chained-run structure, not on the network. A layer outside any kick still costs its
own per-call bracket. With every graph in one kick, each graph under the flag measures at its
`ROCKET_RK3576_I32_SENTINEL=0` minimum [HW sweep, H96 MAX M9, `bench 100`, 2026-08-02].

> [!CAUTION]
> `ROCKET_RK3576_GUARD_PER_KICK=1` detects a corrupted submit that writes nothing. It cannot
> detect a single program that stops writing mid-stream, whose output is then incorrect with no
> error. To detect that case, software must keep the per-layer guard.

The stamp column grows with the residual topology, the placed slices and the pooling layers,
not with the layer count. In the steady state an output keeps its sentinel from the previous
call's verify bracket, at no cost. An output that the chain does not rewrite before it is read
cannot be re-stamped there. It takes a second bracket at the start of the next call instead. Three
kinds of output take one: a placed slice of a concatenation buffer, a declared shared output,
and a pool's output. A pool's handle keeps no flag for the chain to clear.

| | what takes the second bracket |
|---|---|
| MobileNetV1 | no adds at all |
| MobileNetV2 | ten adds |
| ResNet-18 | eight adds plus three shared skip sources plus two pools |
| Inception V1 | 37 placed producers plus thirteen pools |

The stamp column of the table above rises in that order, and Inception V1's stamp is the largest
of the four. It is why wiring Inception V1's concatenations returned 3.26 ms of their saving to
the guard ("Concatenation placement").

### The submit column

The submit column is a round trip plus the execution inside it, because the caller waits on the
fence through the last layer's output. A marginal submit costs between 0.145 and 0.25 ms,
depending on the output it drains. ResNet-18's stem is six row tasks in one submit at 2.13 ms.
With `ROCKET_RK3576_BATCH_TASKS=0` it is six submits at 3.39 ms, 0.25 ms each, on a task
writing a 112×112×64 output. Over twenty small layers, a submit costs 0.145.

A submit also costs less as a graph's submit count falls, as the ResNet-18 and MobileNet
decompositions above measure. Measure a submit's cost on the graph in question rather than
scaling one of these figures. The next section splits the submit column.

### The submit bucket's composition

The submit bucket is the NPU executing programs, and the rest of it is 1-4%. Under the
conditions of the table below, with the guard narrowed, it was 61-81% of each graph's
wall-clock time. It was the largest term on every graph. At the default guard, in the table
above, it is 43-66%. It is one interval covering the ioctl, the kick's execution and the
completion wait, and userspace alone cannot separate those three.

The caller's profile splits the bucket into the submit ioctl, the fence wait and the `FINI_BO`
that hands the output back. The fence wait is a `PREP_BO` with a timeout on the last layer's
output. The driver splits the wait at five events that only it can see:

- Scheduler wakeup: push to `.run_job` entry
- Prologue: `.run_job` to `OPERATION_ENABLE`, which includes `pm_runtime_get_sync`, the IOMMU
  attach test and the first task's register programming
- Execution: the kick to the poll that sees every task retired
- Drain: that poll to the writing block's own completion
- Signal: that completion to `dma_fence_signal`

Per inference, the bucket splits as follows [HW sweep, H96 MAX M9, 2026-08-02]:

| | submit bucket | ioctl | wakeup | prologue | **execution** | drain | signal | `FINI_BO` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV1 | 2.34 ms | 0.057 | 0.012 | 0.003 | **2.252** | 0.001 | 0.008 | 0.004 |
| MobileNetV2 | 2.86 | 0.134 | 0.017 | 0.002 | **2.698** | 0.000 | 0.007 | 0.004 |
| ResNet-18 | 5.84 | 0.055 | 0.014 | 0.002 | **5.752** | 0.000 | 0.008 | 0.004 |
| Inception V1 | 5.56 | 0.107 | 0.018 | 0.003 | **5.450** | 0.001 | 0.008 | 0.004 |

**Conditions:** H96 MAX M9, NPU clock 786 MHz, the RFC series through `0023` with a job-timing
debug module, `bench 100`, guard narrowed, resident weights, cube layout, 2026-08-02. Both 7×7
stems run on the direct lowering: ResNet-18 has 38 tasks and Inception V1 77, where the table
above has 33 and 72.

The terms sum to the measured bucket to under 1%. The driver's own `push -> fence signal` total
matches the caller's fence wait to within tens of microseconds.

Everything other than the NPU executing programs is 0.08-0.17 ms: 1.4-5.8% of the bucket and
1.1-3.7% of the wall-clock time. The submit ioctl is the largest of those terms. It scales with
the BO count rather than the work: ~0.7 us per BO, 0.055 ms for 83 BOs on ResNet-18 and 0.134
for 182 on MobileNetV2. The scheduler wakeup is 12-18 us, the prologue 2-3, the fence signal
about 8 and the `FINI_BO` 4. No dispatch overhead is left to remove from this bucket. A shorter
bucket needs the NPU to do less work, or to do it faster.

The ~340 us per-submit overhead does not appear as a fixed cost inside a one-kick graph. That
figure belongs to a one-task job, where the execution is nil and the overhead is the drain plus
the poll's quantization. A kick costs 0.08-0.17 ms instead.

The `drain` column shows the reason. On every job of all four graphs, the completion bit is
already set on the poll tick that first sees the kick over. An interior program's drain
therefore overlaps the next program's execution. The last program's drain finishes inside the
poll interval that detects the end of the kick.

The drain is a cost of a per-layer graph, not of a chained one. With
`ROCKET_RK3576_NET_KICK=0`, one submit per layer, the same networks spend 0.20-3.17 ms per
inference on it, across 29-72 submits.

The poll period adds at most one period per inference, and tuning it does not change
wall-clock time. Moving `poll_interval_us` from 50 to 10 takes MobileNetV1's execution from
1764 us to 1740 us. The difference is 24 us, half a period, exactly the mean quantization.
Moving it from 50 to 200 costs 145. Neither change moves a wall-clock time outside run-to-run
spread. The execution column is therefore an upper bound on the NPU's own time, by up to one
poll period: 25 us against 2.2-5.8 ms.

The NPU clock does not park. The clocks `aclk_rknn0`, `aclk_rknn1`, `aclk_rknn_cbuf` and
`clk_rknn_dsu0` all read 786431998 Hz idle and under load alike, with no devfreq node. The
RK3576 therefore has no analogue of the RK3588's 200 MHz boot park to undo.

#### The stem's share of the execution

The execution is front-loaded: with the stem on the direct lowering, as in the composition table
above, the stem is a third of it. With one submit per layer
(`ROCKET_RK3576_NET_KICK=0`), each layer has its own profile, with its feature scatter in a
column separate from its submit. The driver's counters give ~0.05 ms of per-job non-execution
in that configuration. Subtracting it from a layer's submit column leaves the layer's execution
plus its own drain [HW sweep, H96 MAX M9, `bench 20`]:

| | stem | its submit | its execution | next layers | graph's execution | stem's share |
|---|---|---:|---:|---:|---:|---:|
| MobileNetV1 | `ic=3 oc=32 224x224 k3 s2` | 0.50 ms | ~0.44 | 0.26-0.47 | 2.25 | **20%** |
| ResNet-18 | `ic=3 oc=64 224x224 k7 s2` | 2.11 | ~2.05 | 0.32-0.39 | 5.75 | **36%** |

Past the stem the layers are flat, at 0.15-0.35 ms of execution each across 28-30 of them.
They sum with the stem to the chained total.

One program of 29 to 81 is a third of ResNet-18's
and Inception V1's whole execution. The stem runs on the direct datapath at the model's own
three input channels. The int8 direct cube is a 32-channel MAC group at every count. The
NPU contracts 32 channels where 3 contain data, so the program does about 10.6x the MACs the
layer needs. Reducing that is a question about the encoding, not about the host or the driver.

The one narrower encoding is the packed-image path. It reaches both 7×7 stem geometries, is
bit-exact there, and is 7.2x faster than the direct stem ("The packed-image encoding at the
stem"). It requires a
non-zero left pad, so it cannot take TFLite's SAME stems.

A chained kick does not make the NPU compute faster. It takes the per-job drain and the poll
quantization off the critical path. The one quantity both configurations measure the same way
is `execution + drain`. A one-task job's `exec` ends at `PC_DONE` and its drain is separate,
while a kick absorbs interior drains into `exec`. On that quantity ResNet-18 takes 6.62 ms per
inference with one submit per layer, against 5.75 chained. The 0.87 ms difference over 30 extra
submits is 29 us each, which is the poll quantization.

With one submit per layer, the drain is 101 us a job on ResNet-18 and 18 on MobileNetV1. It
tracks the bytes each program writes, as the DPU completion's own 82-343 us range implies.

Inside the kick the stem has no profile bucket of its own. Its arithmetic is the same, and it
is the dominant program in that stream.

### The execution cost model

A convolution's execution follows its programmed MAC count, and the feature DMA is 8% of it at
a stem. The direct int8 datapath contracts a 32-channel MAC group at any input-channel count.
It also streams a 32-channel cube through the CBUF, so a narrower encoding would shrink both
terms. The table crosses the MAC count against the feature bytes read, with resident weights
and the driver's measured execution [HW sweep, H96 MAX M9, 2026-08-02]:

| configuration | tasks | programmed MMAC | feature KiB read | execution |
|---|---:|---:|---:|---:|
| **A** `ic=3 oc=64 224x224 s2`, k=1 | 5 | 25.7 | 1533 | 0.301 ms |
| k=3 | 5 | 231.2 | 1596 | 0.452 |
| k=5 | 5 | 642.3 | 1652 | 1.003 |
| k=7 | 6 | 1258.8 | 1743 | 1.802 |
| **B** same MACs, 4.2x fewer bytes: `112x112 k7 s1` | 2 | 1258.8 | 413 | **1.651** |
| **C** k7 `oc=16` (programs a 32-channel group) | 5 | 629.4 | 1708 | 0.953 |
| k7 `oc=32` | 5 | 629.4 | 1708 | 0.952 |
| **D** `k3 oc=32 ic=32` | 5 | 115.6 | 1596 | 0.302 |
| `ic=64` | 9 | 231.2 | 3248 | 0.602 |
| `ic=128` | 19 | 462.4 | 6776 | 1.249 |

Every configuration is bit-exact against a CPU model of the RK3576's arithmetic. Group A holds
the feature cube constant and moves the MAC count 49x. The measured column moves 6.0x, which is
a two-term sum. A least-squares fit to group A gives:

> **execution = 0.218 ms + 1.246e-3 ms per programmed MMAC**: 1.246 ps/MAC, **0.803 TMAC/s**

The fit reproduces the independent output-channel sweep to 0.05 ms, and the two low
input-channel configurations to 0.06 and 0.10. Group B measures the cost of a byte directly. It
holds the MAC count identical and cuts the feature read 4.2x, and the execution goes from 1.802
to 1.651 ms. The feature DMA is therefore 0.151 ms of the stem's execution, 8.4%.

The one configuration the model misses is `ic=128`, by +0.46 ms. Its row allowance collapses to
19 tasks re-reading 6776 KiB. The byte term is small at a stem and grows with the task count,
not with the plane.

In group C, `oc=16` and `oc=32` take the same time to three decimal places. The direct cube
groups output channels by 32, and both program one group. The MAC count that predicts execution
is the programmed one throughout: `round32(oc)`, `round32(ic)`, and every tap of the kernel. A
3-channel stem is expensive for that reason.

#### The packed-image encoding at the stem

At the 7×7 stem the packed-image encoding is 7.2x faster than the direct lowering, from 8x
fewer MACs. The packed-image first conv interleaves four lanes per pixel instead of contracting
a 32-lane group. At the same geometry it therefore programs 8.0x fewer MACs, and it is bit-exact
at both stem geometries [HW sweep, H96 MAX M9]:

| `ic=3 224x224 s2`, in_zp 0 | programmed MMAC | exec | drain | exec+drain | push -> fence signal |
|---|---:|---:|---:|---:|---:|
| k7 `oc=64` direct | 1258.8 | 1.803 ms | 0.000 | 1.803 | 1.856 ms |
| k7 `oc=64` **packed** | 157.4 | 0.053 | 0.198 | **0.251** | **0.290** |
| k3 `oc=32` direct | 115.6 | 0.302 | 0.000 | 0.302 | 0.344 |
| k3 `oc=32` **packed** | 14.5 | 0.053 | 0.049 | **0.102** | **0.138** |

Compare `exec + drain`, not `exec`. A short program's drain is exposed, where a long one's
finishes inside the poll interval that detects the job over. On `exec + drain` the packed form
is 7.2x faster than the direct form at the 7×7 stem and 3.0x at the 3×3. At the 7×7 geometry it
saves 1.566 ms of driver-measured job time per inference. The two encodings run at the same
rate, 1.59e-3 against 1.43e-3 ms per programmed MMAC. The packed path saves time by programming fewer MACs, not
through a faster datapath, as the cost model predicts.

#### The packed encoding's bounds on the four graphs

The packed-image encoding has four decoded bounds:

- A non-zero left pad
- An output width of `iw/stride`
- An output width that is a multiple of 16
- For an exact border, a zero input zero point

The last bound is a hardware one. On the packed encoding the pad columns feed 0 at both ends,
whatever `CNA_PAD_CON1` holds.

> [!CAUTION]
> With a non-zero input zero point, a packed program's pad columns hold 0 rather than the zero
> point, and its border outputs are incorrect. Software must use a zero input zero point, or
> extend the image at the zero point and crop the surplus output column.

What blocks the encoding differs on each graph. The four graphs' stems against those bounds:

| graph | stem | `pad_left` | `in_zp` | packed encoding |
|---|---|---:|---:|---|
| MobileNetV1 | `ic=3 oc=32 k3 s2` | 0 | 0 | refused: TFLite SAME has no leading pad |
| MobileNetV2 | `ic=3 oc=32 k3 s2` | 0 | 0 | refused: same |
| ResNet-18 | `ic=3 oc=64 k7 s2` | 3 | **-128** | **runs**, with the pad columns materialized |
| Inception V1 | `ic=3 oc=64 k7 s2` | **2** | **0** | **runs** |

The encoding requires a leading pad rather than refusing one. That bound is the RK3576's, and it
blocks both MobileNets ([../chips/rk3576.md](../chips/rk3576.md) §"First convolution on a packed
image"). Their stems reach the packed path only through an input shift, which costs more than it
saves in a resident graph ([../chips/rk3576.md](../chips/rk3576.md) §"TFLite stem through an
input shift").

ResNet-18's zero point meets the measured hardware bound above. Repairing it on the host costs
9.8 ms against a ~2 ms saving. The columns are materializable, though. Extending the image at
the zero point and cropping the surplus output column is bit-exact against the direct lowering
on that exact stem. The bound is therefore on the naive descriptor rather than on the encoding.

`librocketnpu` performs the extension. In the graph the stem runs packed, because its consumer,
the max pool, reads the wider output in place ([../chips/rk3576.md](../chips/rk3576.md) §"The
materialized stem on a graph"). The MobileNets' refusals in the table are `librocketnpu`'s own
messages, from a graph run.

#### The asymmetric output extent on the packed encoding

The CNA takes no configured trailing pad. It derives the pad its last window consumes from the
output extent and the leading pad. The emitter writes that derivation into `CNA_PAD_CON0`'s high
bytes for both sub-encodings, from one shared `r76_trail_pad(OW, stride, KW, pad_left, IW)`. The
packed converter reads those bytes exactly as the direct path does. The packed encoding
therefore honors an asymmetric output extent [HW sweep, H96 MAX M9]. Eight asymmetric test
points are bit-exact, over five axes:

- Kernels 5 and 7
- Strides 1 and 2
- Planes 64/128/224
- Trailing pads 2/3/4
- Each axis asymmetric while the other stays derived

The bound on an extent on this path is the shared rule every int8 entry applies. A trailing pad
of `kh` or more is a whole window of pad.

#### A packed program at the head of a chained stream

A packed program can head a chained stream. It chains over five geometries, with both of its
boundaries scored [HW sweep]. All four graphs are one hardware kick in one submit. A stem that
cannot chain splits the chained run and adds a 784 KiB join, measured at 6.80 ms. That join would make
the cheaper program a large net loss.

The packed sub-encoding is the one place a convolution program configures the CNA front end
differently. It has a folded channel count that disagrees with the DMA's, a live CVT stage, and
a packed row stride instead of a cube's. Any of that crossing a chained boundary would show in
the program after it, and none does.

A packed handle can only begin a chained run. Its feature buffer is an interleaved image, and no
producer writes one. Beginning a chained run is the only position a stem occupies anyway. Its output
side is an ordinary NC1HWC2 cube and joins normally.

#### Residency and the vectorized image pack

The packed stem's saving appears once the encoding is resident and its pack is vectorized. The
table measures the whole prepacked entry, resident in both configurations, with the governor
pinned [HW sweep, H96 MAX M9]:

| `ic=3 224x224 s2`, resident | tasks | programmed MMAC | packed | direct | packed saves |
|---|---:|---:|---:|---:|---:|
| Inception V1 stem, k7 `oc=64` pad 2, extent 112 | 1 | 157.4 | **1.24 ms** | 3.09 | **1.85 ms (60%)** |
| k7 `oc=64` pad 3 | 1 | 157.4 | **1.23** | 3.12 | **1.89 (61%)** |
| k3 `oc=32` pad 1 | 1 | 14.5 | **0.63** | 1.13 | **0.50 (44%)** |

The packed program plans into one row task where the direct lowering takes five or six, and
every configuration is bit-exact against the CPU model.

Two library properties provide the saving, each worth about a millisecond:

- **Residency.** Through the transient entry, the packed encoding incurs a per-call BO teardown
  that the resident direct configurations do not. Profiled transiently at the Inception V1
  geometry, the call was 4.79 ms, of which teardown was 1.82 and `plan+gen` 0.85. One handle
  serves both encodings: `rocket_conv2d_int8_pack_rk3576()` packs the packed cube as well as the
  direct one. For a transient call, `rocket_conv2d_int8_rk3576()` builds the same handle, so
  there is no second arithmetic to keep in agreement. The per-axis requant and the fused LUT
  activation are refused on it. Both are direct-path options, and neither is measured on this
  encoding.
- **The host image pack.** Converting a CHW tensor to an interleaved image is a 2/3/4-way
  interleave of whole planes, one `vst2`/`vst3`/`vst4`, not the cube path's 16×16 block
  transpose. Written an element at a time, the pack gives most of the saving back. At the
  Inception V1 stem the wall-clock time is 2.80 ms scalar against 1.64 vectorized. The encoding
  saves 1.70 ms of submit, and the scalar form took 1.88 ms on that loop. The shared entry is
  `rocket_rk3576_argb_pack()`.

> [!NOTE]
> The H96 MAX M9 runs the `ondemand` CPU governor. A host-bound configuration's wall-clock time
> then depends on what ran before it in the same process. Pin the governor for a comparison
> whose configurations differ in host work. Discard the first process as well as the first
> iteration. With `performance` set, runs 2 and 3 agree to 0.01 ms while run 1 is 25% high.

#### What the packed form adds and removes

Against the direct lowering, the packed form removes the MAC count and five row tasks. It adds
these:

- A host interleave, which is vectorized
- A weight cube that is a different object
- Four geometry bounds, three of them silent if violated, all refused
- A zero-input-zero-point bound
- No per-axis requant
- No fused activation
- No cube in

None of it changes which graphs can use the encoding. It runs on two graphs of four, Inception
V1 directly and ResNet-18 with its pad columns materialized. The bound that blocks both
MobileNets, a non-zero left pad, is the RK3576's.

#### The packed stem in the Inception V1 graph

The packed-image encoding is bit-exact at a non-zero weight zero point. Inception V1's stem has
`w_zp = 13`, and every layer of the graph is exact against the CPU model. The packed path's tap
count is the live `ic*kh*kw`, and its feature buffer has no padded MAC group. The `B` term
therefore folds over exactly the taps the program walks.

In the graph, the packed stem saves 2.0-2.1 ms per inference, 19% of Inception V1's wall-clock
time at the default guard. The result is byte-identical, in the same one submit and one kick
over 81 layers [HW sweep, H96 MAX M9, `bench 100`]. The task count goes 77 -> 72. The
saving is the same 2.1 ms with `GUARD_NARROW=1` (27%) and with `GUARD_PER_KICK=1` (34%), since
the guard does not move: 122 brackets over 14445 KiB either way.

Neither MobileNet changes, and `librocketnpu` prints its refusal for each of their stems.
ResNet-18's stem takes the packed encoding too, with its pad columns materialized
([../chips/rk3576.md](../chips/rk3576.md) §"The materialized stem on a graph").

The profile (`ROCKET_RK3576_INT8_PROF=1`) puts the saving in two terms:

| term | direct | packed | delta |
|---|---:|---:|---:|
| submit (the NPU executing programs) | 5.59 ms | 4.05-4.15 | **-1.48** |
| feature (the host operand pack) | 0.50-0.55 | 0.07-0.09 | **-0.44** |
| stamp + verify (the write guard) | 4.28-4.45 | 4.36-4.63 | 0 |
| de-tile | 0.01 | 0.01 | 0 |

The feature term is the second half of the saving. The direct lowering scatters the image into
a 32-channel cube, `round32(3)`, and the packed one interleaves three bytes per pixel instead.
The interleave moves about 10x fewer bytes, through a loop that is vectorized on both paths.

A per-op comparison, such as the 1.85 ms in the residency table, does not bound the in-graph
saving. Each per-op call includes `plan+gen` and a readback that the chain does not, so the two are
different quantities.

The packed handle's output is the next layer's cube, as measured on the graph. The graph keeps
all 53 joins of 80 adjacent pairs and its single 81-layer chained run, so the stem is an ordinary
interior-adjacent producer.

## Evidence

The method, controls, excluded causes and run detail behind each section are in
[../chips/rk3576-validation.md](../chips/rk3576-validation.md), under the same titles. They run
from §"A whole network" to §"Composition of a graph's wall-clock time".
