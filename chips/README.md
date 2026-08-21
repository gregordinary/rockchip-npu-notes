# Per-SoC parameter sheets

The rest of `rockchip-npu-notes` describes the RK3588 NPU. Most of it is
**IP-inherent** — true of the rknpu/NVDLA-derived block on any Rockchip SoC that
carries it: the matmul-as-1×1-convolution model, the CNA→CORE→DPU sequence, the
precision-field encodings, the BS/BN/EW/LUT datapath, the tile-layout algebra, the
"no on-chip layout conversion" and "no hardware gather" facts. Those do not change
from chip to chip and are stated once, in the subsystem docs.

What **does** change per SoC lives here, one sheet per chip:

- **Machine parameters** — CBUF banks and bank size, the matmul tile cap, the
  weight tile-group sizes, the usable-datatype menu, the core count. These are the
  values `rocket-userspace`'s `rocket_hw_profile` collects.
- **The register offset map** — the numeric CNA/CORE/DPU register offsets. These are
  IP-**revision**-specific, not shared across every Rockchip NPU: the RK3576 NPU
  re-packs the CNA block relative to the RK3588 (see [rk3576.md](rk3576.md)). The
  *block bases* (CNA at core+0x1000, CORE at +0x3000) and the register *semantics*
  are shared; the within-block offsets are not.
- **SoC integration** — physical base addresses, clocks, resets, power domains,
  IRQs, IOMMU wiring. Pure device-map facts.

| SoC | NPU cores | Status | Sheet |
|---|---|---|---|
| RK3588 | 3 | Hardware-validated (the reference target) | [rk3588.md](rk3588.md) |
| RK3576 | 2 | Runs on hardware. Its int8 **direct** conv encoder computes bit-exactly across a swept geometry envelope, with every boundary closed by construction. Its fp16 convolution computes bit-exactly too and delivers every output channel — `rocket_rk3576_plan_ic()` splits an arbitrary `ic` into the 16-channel slices one task contracts, bit-exact against a CPU model at `ic` 16/32/64/128, `k` 1/3/5, planes to 56x56 — at `ic/16` submits of ~1.4 ms each. One bound sets the envelope: the DPU's output element stride is `16/ic` words, so 16 is the contraction width at which an element lands in its own two bytes. Its **depthwise** conv computes bit-exactly too, over channel counts 8-256 (including non-multiples of 32), kernels 1-7, both strides and odd planes — its registers came from manufactured captures, but its coefficient group and int8 weight cube are neither the direct path's nor in any capture and had to be read off the part. **Matmul** computes at int8 through `rocket_matmul_int8_rk3576()`, whose output is int8 through the DPU's requant. K past one task's contraction is SPLIT, through the 32-bit writer: its budget is one 16-byte atom per (16-channel block, pixel) whatever the OUTPUT element width is, so most channels never reach DDR, and a wider programmed `oc` plus a weight-cube scatter turn the surface back into a plain cube — bit-exact to K=16384. What the budget IS a function of is the DPU's own operand width, and widening it halves the scatter to 2x, which doubles the N tile and halves the submits — decoded and gated, but opt-in, because that mode intermittently emits a contiguous block of its stream as zeros; a sentinel stamped into the surface shows the writer reaches those atoms rather than skipping them, and the block is contiguous in the writer's emission order, so it is detected and the row task redone. An i32out job poisons the next submit, of any kind and across processes, and what clears it is the driver's runtime-PM autosuspend cycling the NPU power domain rather than elapsed time — so the cost is settable from `power/autosuspend_delay_ms`; the plain int8 matmul inherits that hazard too and now detects it the same way. The coefficient group's asymmetric `B` term is settled: the DPU ADDS `B*sum(x)`, so a weight zero point is programmed negated. The rest of the op library has no encoder for this part and REFUSES rather than emitting the RK3588 program, which this part completes without writing anything | [rk3576.md](rk3576.md), [rk3576-regcmd.md](rk3576-regcmd.md) |
| RK3566 | 1 | Planned (hardware incoming) | [rk3566.md](rk3566.md) |

A chip can need two sheets. The RK3576's parameter sheet covers identity, SoC
integration and the state of the port; its register encoding is large enough — and
concrete enough now that it is an implemented emitter with a gate — to live separately
in [rk3576-regcmd.md](rk3576-regcmd.md).

An IP-inherent fact that turns out to differ on a new chip is a bug in the claim, not
a per-SoC parameter — promote the correction into the subsystem doc and note the SoC
that disproved the old scope. The RK3576 CNA re-pack did exactly that to the old
"register offsets are not SoC-specific" claim.
