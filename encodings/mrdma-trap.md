# The MRDMA trap (DPU_RDMA)

The DPU's read DMA, [DPU_RDMA](../GLOSSARY.md), has two sources: MRDMA, the main feed, and
ERDMA, the eltwise operand feed. `DPU_RDMA_FEATURE_MODE_CFG` (`0x5044`) selects them through
two bits, `flying_mode` and `mrdma_disable`. A plain convolution or matmul takes the DPU's main
data from the convolution output, and its MRDMA must be disabled. An enabled MRDMA with nothing
to read hangs the job (Erratum E1).

## Registers

| Offset | Field | Bits | Reset | Function |
|---|---|---|---|---|
| `0x5044` `DPU_RDMA_FEATURE_MODE_CFG` | `flying_mode` | 0 | 0 | Selects the DPU main data: 0 = from the convolution output, 1 = from MRDMA |
| `0x5044` `DPU_RDMA_FEATURE_MODE_CFG` | `mrdma_disable` | 4 | 0 | 1 = MRDMA disabled |

## Programming requirements

The DPU_RDMA block must match the operation the task runs:

| Operation | DPU_RDMA programming |
|---|---|
| Plain convolution or matmul | `mrdma_disable` = 1, ERDMA disabled |
| Eltwise op or fp16 K-accumulation | MRDMA enabled and fed (`COMB_USE(5)`), ERDMA armed ([k-accumulation.md](k-accumulation.md)) |
| Two-buffer EW op, no convolution | `flying_mode` = 1, MRDMA fed from memory, ERDMA armed, `COMB_USE` bit 0 clear ([sdp-stage-precision.md](sdp-stage-precision.md)) |

For a plain convolution or matmul, software must emit the DPU_RDMA block (`0x5xxx`) with
`mrdma_disable=1`. Software must also include the DPU_RDMA bit in the enable mask,
`PC_OPERATION_ENABLE` ([regcmd-task-model.md](regcmd-task-model.md)). Mesa uses `0x1D`, bits
0, 2, 3 and 4, which is one more block than a mask without DPU_RDMA.

> [!CAUTION]
> A plain convolution or matmul programmed with the eltwise path's DPU_RDMA block hangs the
> same way as Erratum E1. With the ERDMA armed in `DPU_RDMA_ERDMA_CFG` and the MRDMA enabled
> with `COMB_USE(5)` in `DPU_RDMA_FEATURE_MODE_CFG`, the job times out with the output
> untouched.

## Software support

A register generator whose DPU descriptor has an `ew_accumulate` field must set it to 0
explicitly for the plain path. A descriptor that leaves `ew_accumulate` uninitialized reads
stack garbage. A nonzero value selects the ERDMA-armed eltwise path and produces the same
"timed out, output untouched" hang. The fix is one line: `dpu_desc.ew_accumulate = 0;`.

## Known issues

### Erratum E1: job timeout with the output untouched

- **Description:** With `mrdma_disable` clear, MRDMA is enabled and has no data to read. The
  DPU read DMA waits for data that never arrives, and the task never finishes [HW sweep,
  source-confirmed].
- **Conditions:** RK3588 under the `rocket` driver. A task that runs a plain convolution or
  matmul with no eltwise operand, and leaves `mrdma_disable` clear. A register program that
  configures CNA, CORE and DPU and never writes the `0x5xxx` block leaves
  `DPU_RDMA_FEATURE_MODE_CFG` at its default 0, and meets this condition.
- **Symptom:** The job times out, and the output BO is untouched. The `PREP_BO` deadline
  expires (`WAIT TIMEOUT -110`). There is no IOMMU fault and no kernel panic [HW sweep].
- **Workaround:** Software must program `mrdma_disable=1` in `DPU_RDMA_FEATURE_MODE_CFG`, and
  must include the DPU_RDMA bit in `PC_OPERATION_ENABLE`. Mesa's no-eltwise path does both
  [source-confirmed: Mesa `rkt_regcmd.c`]. With the `0x5xxx` block emitted this way, an fp16
  matmul on `rocket` produces correct output.
- **Status:** Open.

### Recovery from a hang

> [!CAUTION]
> A wrong eltwise or K-accumulation configuration can leave the NPU hung. The next submit
> also times out, even after the configuration is fixed. When you sweep DPU_RDMA geometries
> that can hang, reload the `rocket` module (`rmmod`, then `insmod`) after each failing
> attempt.

## Evidence

### Known issues

The hardware symptom and Mesa's `rkt_regcmd.c` both confirm the cause [HW sweep,
source-confirmed]. Mesa always emits the DPU_RDMA block, with `mrdma_disable=1` on the
no-eltwise path.
