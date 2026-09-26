# The NVDLA ancestor and its documentation

The Rockchip NPU is an NVDLA derivative, and NVDLA is documented in the open
([nvdla.org](http://nvdla.org/), hardware manual v1). That documentation is the only
authoritative prose about this datapath. Rockchip publishes a register list in the TRM and
nothing about semantics. So it is worth knowing exactly which questions it answers,
which it answers *wrongly* for this silicon, and which it cannot reach at all.

The short version: **NVDLA is a floor, not a ceiling.** The structure (blocks, buffers,
pipeline stages, register-file organization) transfers. The arithmetic details do not
always. And several of the fields that matter most here are Rockchip additions NVDLA never
had, so its silence about them means nothing.

## Block map

| NVDLA | RK block | base | note |
|---|---|---|---|
| CDMA (convolution DMA) | CNA | `0x1000` | one block on the RK; NVDLA splits DC / WG / IMG / WT engines inside it |
| CBUF (convolution buffer) | n/a | n/a | no registers of its own in NVDLA; on the RK it is configured from CNA `0x1040`/`0x103c` |
| CSC (sequence controller) | CNA | `0x1000` | `CONV_CON2.CSC_DO_EN` / `CSC_WO_EN` are its data/weight loaders |
| CMAC + CACC | CORE | `0x3000` | `MAC_GATING`, `MISC_CFG`, `CLIP_TRUNCATE` |
| SDP + SDP_RDMA | DPU + DPU_RDMA | `0x4000` / `0x5000` | BS/BN/EW == NVDLA X1/X2/Y; the LUT is NVDLA's hybrid LE/LO |
| PDP + PDP_RDMA | PPU + PPU_RDMA | `0x6000` / `0x7000` | planar pooling |
| MCIF (memory interface) | DDMA | `0x8000` | arbitration / QoS / outstanding, see below |
| SRAMIF (second memory bus) | SDMA | `0x9000` | the on-chip-SRAM port |
| GLB | global | `0xF008` | one `OPERATION_ENABLE` per block |
| CDP (cross-channel LRN) | **absent** | n/a | no channel-axis datapath; feature-axis reduction is a ones-matmul |
| BDMA (bridge DMA) | **absent** | n/a | no memory-to-memory copy engine |
| RUBIK (reshape engine) | **absent** | n/a | the reason host de-tiling is irreducible |
| n/a | **PC** | `0x0000` | Rockchip's own: a register-program fetcher, so a "hardware layer" is a memory buffer instead of a CPU write sequence |

The three absences are load-bearing negatives elsewhere in these notes. No RUBIK is why
[the host packing cannot be moved on-chip](perf/ppu-pooling-not-detile.md). No CDP is why
[reduction over the feature axis is a matmul](encodings/feature-reduce.md).

## What transfers, and is worth reading

**The ping-pong register file** [source-confirmed]. Every NVDLA sub-unit has three register
groups: two duplicated per-hardware-layer groups sharing one address range, and one
non-shadowed group holding status and pointers. `PRODUCER` selects the group the CPU writes.
`CONSUMER` is read-only, and says which group the datapath is sourcing. Hardware
clears the running group's enable bit at layer end, advances `CONSUMER`, and starts
immediately if the other group's enable is already set. **Writes to a group whose enable
bit is set are dropped silently**: there is no error, and software cannot clear the bit.

That maps onto the RK registers exactly:

| NVDLA | RK |
|---|---|
| `POINTER.PRODUCER` | `S_POINTER[0]` `POINTER` |
| `POINTER.CONSUMER` | `S_POINTER[16]` `EXECUTER` |
| `S_STATUS` per-group state | `S_STATUS[1:0]` `STATUS_0`, `[17:16]` `STATUS_1` |
| `D_OP_ENABLE` | `OPERATION_ENABLE[0]` `OP_EN` |

`S_POINTER` sits at `0x1004`, `0x3004`, `0x4004`, `0x5004`, `0x6004` and `0x7004`. It also
carries `POINTER_PP_EN`, `EXECUTER_PP_EN`, `POINTER_PP_MODE` and the two `PP_CLEAR` bits, which
are Rockchip's automation of the advance. Every emitter in this tree and in Mesa writes `0xE`
(`PP_MODE | EXECUTER_PP_EN | POINTER_PP_EN`) and lets the hardware flip the pointers. It is
also why the PC interrupt bits come in pairs (`CNA_FEATURE_0/1`, `CNA_WEIGHT_0/1`,
`CNA_CSC_0/1`, `CORE_0/1`, `DPU_0/1`, `PPU_0/1`): the suffix is the register **group**, not
a core.

NVDLA also states plainly what our task model has to respect: **the hardware does no
dependency tracking between layers**. Two pending layers cannot block each other, so if one
consumes the other's output, software must not schedule it until the producer has finished.

**The convolution buffer** [source-confirmed]. 16 banks act as **three** logical circular
buffers: input data, weight, and WMB compression tags. In the v1 implementation it has **two
write ports and three read ports**.

The architectural spec is blunter still, and requires four. Those are a "read port for feature
data, read port for weight data, **write port for feature data, write port for weight data**".
Bank 15 is reserved for WMB when weights are compressed. Otherwise data and weight share all 16.

Two things follow for us. First, the allocation is by **bank count** from the bottom. NVDLA has
no CBUF base register at all, and a layer stages implicitly from zero. The RK's `CNA_CBUF_CON0`
is that count, plus a granule *allowance* on the RK3576. Neither is an address partition, which
is the architectural reason [no partition was found on the RK3576](chips/rk3576.md).

Second, the feature-write path is a **separate, single port**. That is what the surviving RK3576
hypothesis rests on. Contention there predicts extent proportional to the aggressor's feature
plane, indifference to weight bytes and to occupancy, and immunity to placement. All three are
measured.

**The accumulator** [source-confirmed]. CACC accumulates partial sums across the stripe loop
**within one hardware layer**, in an assembly SRAM at INT34, INT48 or FP44. It explicitly zeroes
at the first stripe. It rounds and saturates to INT32/FP32 before the
SDP, under a register NVDLA calls `D_CLIP_CFG.CLIP_TRUNCATE`, the RK's `CORE_CLIP_TRUNCATE`
(`0x301C`). There is no cross-layer accumulate, which is the independent confirmation of
[the K-accumulation ceiling](encodings/sdp-stage-precision.md).

**The DMA arbiters** (MCIF / SRAMIF -> DDMA / SDMA). Five read clients with per-client
arbitration weights and 2-bit QoS (`FEATURE`, `KERNEL`, `DPU`, `PDP`, `PC`), plus
outstanding-transaction limits, a round-robin/fixed arbiter select, and the AXI attributes.
Nothing in this stack or in Mesa ever writes them. Read on an idle RK3588 core 0 they are
`CFG_OUTSTANDING = 0x00000fff` (read outstanding already at the 8-bit maximum 255, write at
15), `RD_WEIGHT_0 = 0x01010101` and `RD_WEIGHT_1 = 0x00000101`, with every client weighted
equally [HW, see [perf/hw-byte-counters.md](perf/hw-byte-counters.md)]. So the read path is
not outstanding-limited at the default, and the only untried knobs are the relative weights
and the write outstanding count.

## What does not transfer

**The rounding rule, and this is the sharp one.** NVDLA specifies that both the convertor and
the truncate stage round **half away from zero**. The RK parts add a select, `OUT_CVT_SHIFT[30]`.
Every program here and in the vendor's corpus clears it, which rounds **half to even**
(0.5 -> 0, 1.5 -> 2, −0.5 -> 0, −1.5 -> −2), and setting it gives NVDLA's rule. Both settings are
measured on the RK3576 and the RK3588, and the RK3576's BS-stage shift also rounds half to even
[HW sweep, `tests/requant_round_probe.c`, `tests/rk3576_coeff_c.c`]. See
[encodings/out-cvt-converter.md](encodings/out-cvt-converter.md) for what that costs and why no
gate had ever exercised it.

Inherited datapath semantics are worth a **prediction** rather than a conclusion. This one was
worth running precisely because it was cheap to check and the documentation was confident.

**Anything Rockchip added.** The register map carries fields with no NVDLA counterpart, so the
ancestor's silence about them is not evidence:

- `PC` (`0x0000`): the register-program fetcher. NVDLA is programmed by a CPU over its CSB. The
  RK streams a regcmd list from memory, which is the whole basis of `rocket`'s uAPI.
- `CNA_CONV_CON1[16]` `DECONV` plus `CONV_CON3[13:11]/[10:8]` `DECONV_Y/X_STRIDE`: a
  **transposed-convolution mode**. NVDLA has no deconvolution engine at any revision. It is live
  on the RK3588, in [encodings/conv-transpose.md](encodings/conv-transpose.md).
- `DPU_RDMA` `0x5024` (`BS_BASE_ADDR1`): an address the DPU reads a per-sign shift word through,
  `[5:0]` for non-negative results and `[13:8]` for negative. It is decoded on the RK3576, absent
  from Mesa's register map, and never written by the RK3588 emitters.
- The RK3576's CBUF granule allowance in `CNA_CBUF_ENTRIES` / `CNA_CBUF_CON0`.

**Where the two chips disagree with each other**, NVDLA cannot arbitrate. The RK3576's
wide-output poisoning, its missing DPU completion for wide output elements, and its two-core
corruption are all integration behavior. None has a counterpart in a single-core reference
design.

## Features NVDLA has that we have deliberately not chased

**Winograd** (`CDMA_WG`). NVDLA runs 3×3 stride-1 convolutions through a Winograd transform for
2.25x fewer MACs. That needs a dedicated fetch engine, its own CBUF layout, mandatory channel
extension, and an extra `pra_trunc` precision stage.

The RK's `CNA_CONV_CON1.CONV_MODE` is a 4-bit field of which only `0`, direct, is known, so there
is room for it.

It is not worth chasing here. [The matmul is DMA/dispatch-bound rather than
MAC-bound](perf/not-mac-bound.md) at this operating point, so fewer MACs buy nothing. The cost is
an entirely new host cube layout and more CBUF pressure. Revisit only if the dispatch floor comes down far
enough to make MACs the constraint.

**Multi-batch** (`CDMA_DC` only, up to 32 cubes per layer, `BATCH_STRIDE`). NVDLA's pitch is
fully-connected layers: load the weights once, stream many activation sets, for "overall
performance close to [batch] × [single-layer performance]".

That is exactly the shape of the LLM decode problem. But in our formulation a matmul's M **is**
the batch axis, because M is the conv's spatial height. Weight reuse across M is already what we
do, and
[decode is slow for a different reason](perf/decode-gemv.md). The one part that would be new
is `BATCH_STRIDE`: NVDLA's batch cubes live at a fixed stride in memory rather than
contiguously. No such field is identified in the RK map.

## Reading list

Fetch and read as text rather than skimming the site. The pages are large, and the useful parts
are deep in them.

| page | what is in it |
|---|---|
| `nvdla.org/hw/v1/hwarch.html` | the ping-pong mechanism, CBUF ports, per-block register tables, the DBBIF/SRAMIF interfaces |
| `nvdla.org/hw/v1/ias/unit_description.html` | per-unit behavior: CDMA's four engines, CBUF banking, CSC, CMAC, CACC precision |
| `nvdla.org/hw/v1/ias/precision.html` | the convertor / truncate / shifter formulas, the rounding rule, the zero-point-into-padding derivation |
| `nvdla.org/hw/v1/ias/programming_guide.html` | per-block programming order and constraints (`op_enable` in reverse pipeline order, bank sizing rules) |
| `nvdla.org/hw/format.html` | feature/weight cube formats and the compression tag layout |
| `nvdla.org/hw/v1/ias/lut-programming.html` | the LE/LO hybrid LUT this stack's activations use |

One thing from the precision page is worth quoting, because we already do it and it is easy to
get wrong. With an input zero point, convolution padding must be **valued** rather than zero:
`PADDING_VALUE = −offset·SF`. Our encoders fold the input zero point into `CNA_PAD_CON1` for
exactly this reason.
