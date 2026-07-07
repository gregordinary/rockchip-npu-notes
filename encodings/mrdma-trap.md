# The MRDMA trap (DPU-RDMA)

**Symptom:** you emit a regcmd that configures CNA / CORE / DPU correctly, submit the
job, and it **times out** (`PREP_BO` deadline expires, `WAIT TIMEOUT -110`) with the
output BO **untouched**. No IOMMU fault, no kernel panic — the device just never
finishes.

**Cause:** the DPU's read DMA has two sources — **MRDMA** (the main feed) and
**ERDMA** (the eltwise operand feed) — governed by `DPU_RDMA_FEATURE_MODE_CFG`
(`0x5044`): bit0 `flying_mode` (0 = DPU main data comes from the conv output, 1 = from
MRDMA) and bit4 `mrdma_disable` (1 = disable MRDMA). For a plain convolution/matmul
with **no** eltwise add, the conv output is the DPU's main data, so **MRDMA must be
disabled.** If you leave `DPU_RDMA_FEATURE_MODE_CFG` at its default 0, `mrdma_disable`
is unset → MRDMA is enabled but **unfed** → the DPU read-DMA waits forever for data
that never arrives → the job times out.

This is confirmed by the HW symptom above and by Mesa's `rkt_regcmd.c`, which
**always** emits the DPU-RDMA block (with `mrdma_disable=1` on the no-eltwise path).
[HW + source-confirmed]

## The fix

For a no-eltwise matmul, emit the **DPU-RDMA block (`0x5xxx`) with `mrdma_disable=1`**
(Mesa's no-eltwise path) and include the DPU-RDMA bit in the enable mask (Mesa uses
`0x1D` = bits 0,2,3,4 — one more block than a config that forgets DPU-RDMA). A generator
that configures CNA/CORE/DPU but never writes the `0x5xxx` domain at all produces the
hang; emitting the `0x5xxx` block with `mrdma_disable=1` is what makes a correct fp16
matmul on `rocket`.

## The corollary trap (eltwise path)

The mirror-image hang hits the eltwise path. When you reuse a generator that has an
`ew_accumulate` field, you must set it **explicitly to 0** for the plain path. A
descriptor that leaves `ew_accumulate` uninitialized reads stack garbage — a nonzero
value routes to the ERDMA-armed eltwise path and reproduces the exact same
"timed out, output untouched" hang. **One line:** `dpu_desc.ew_accumulate = 0;`.

So the rule is symmetric:

- **plain matmul** → `mrdma_disable = 1`, `ew_accumulate = 0`.
- **eltwise / fp16 K-accum** → MRDMA *enabled* and fed (`COMB_USE(5)`), ERDMA armed
  (see [k-accumulation.md](k-accumulation.md)).

Get the DPU-RDMA block wrong in *either* direction and the job silently times out
instead of erroring usefully — which is what makes this a trap rather than a bug.

## Recovery note

A wrong eltwise/K-accum config can **wedge the NPU**: the timeout carries over to the
next submit even after you fix the config. When sweeping risky DPU-RDMA geometries,
`rmmod`/`insmod` the `rocket` module between failing attempts to clear the wedge.
