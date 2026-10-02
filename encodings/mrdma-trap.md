# The MRDMA trap (DPU-RDMA)

**Symptom:** a regcmd configures CNA, CORE and DPU correctly, and the submitted job times
out with the output BO untouched. The `PREP_BO` deadline expires (`WAIT TIMEOUT -110`).
There is no IOMMU fault and no kernel panic. The device never finishes.

**Cause:** the DPU's read DMA has two sources, MRDMA (the main feed) and ERDMA (the
eltwise operand feed). `DPU_RDMA_FEATURE_MODE_CFG` (`0x5044`) governs them through two
bits. Bit0 `flying_mode` selects the DPU main data (0 = DPU main data comes from the conv
output, 1 = from MRDMA). Bit4 is `mrdma_disable` (1 = disable MRDMA).

For a plain convolution or matmul with no eltwise add, the conv output is the DPU's main
data, so **MRDMA must be disabled**. If `DPU_RDMA_FEATURE_MODE_CFG` stays at its default 0,
`mrdma_disable` is unset, and MRDMA is enabled but unfed. The DPU read DMA then waits
forever for data that never arrives, and the job times out.

The hardware symptom above and Mesa's `rkt_regcmd.c` both confirm the cause
[HW sweep, source-confirmed]. Mesa always emits the DPU-RDMA block, with
`mrdma_disable=1` on the no-eltwise path.

## The fix

For a no-eltwise matmul, emit the DPU-RDMA block (`0x5xxx`) with `mrdma_disable=1`, which
is Mesa's no-eltwise path. Include the DPU-RDMA bit in the enable mask. Mesa uses `0x1D` =
bits 0,2,3,4, one more block than a config that forgets DPU-RDMA. A generator that
configures CNA, CORE and DPU but never writes the `0x5xxx` domain produces the hang.
Emitting the `0x5xxx` block with `mrdma_disable=1` is what makes a correct fp16 matmul on
`rocket`.

## The corollary trap (eltwise path)

The mirror-image hang hits the eltwise path. When you reuse a generator with an
`ew_accumulate` field, you must set it explicitly to 0 for the plain path. A descriptor
that leaves `ew_accumulate` uninitialized reads stack garbage. A nonzero value routes to
the ERDMA-armed eltwise path and reproduces the same "timed out, output untouched" hang.
The fix is one line: `dpu_desc.ew_accumulate = 0;`.

So the DPU-RDMA block must match the op in both directions:

- **Plain matmul**: `mrdma_disable = 1`, `ew_accumulate = 0`
- **Eltwise or fp16 K-accumulation**: MRDMA enabled and fed (`COMB_USE(5)`), ERDMA armed
  (see [k-accumulation.md](k-accumulation.md))
- **Two-buffer EW op, no conv**: `flying_mode` 1, MRDMA fed from memory, ERDMA armed,
  `COMB_USE` bit 0 clear (see [sdp-stage-precision.md](sdp-stage-precision.md))

With the DPU-RDMA block wrong in either direction, the job times out silently and raises
no useful error. That silence is what makes this a trap rather than a bug.

## Recovery note

A wrong eltwise or K-accumulation config can **wedge the NPU**: the timeout carries over to
the next submit even after you fix the config. When sweeping risky DPU-RDMA geometries,
reload the `rocket` module (`rmmod`, then `insmod`) between failing attempts to clear the
wedge.
