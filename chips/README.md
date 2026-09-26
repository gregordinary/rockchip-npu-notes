# Per-SoC parameter sheets

The rest of `rockchip-npu-notes` describes the RK3588 NPU. Most of it is **IP-inherent**, meaning
true of the rknpu and NVDLA-derived block on any Rockchip SoC that carries it:

- The matmul-as-1×1-convolution model.
- The CNA→CORE→DPU sequence.
- The precision-field encodings.
- The BS/BN/EW/LUT datapath.
- The tile-layout algebra.
- The "no on-chip layout conversion" and "no hardware gather" facts.

Those do not change from chip to chip, and are stated once in the subsystem docs.

What **does** change per SoC lives here, one sheet per chip:

- **Machine parameters**: CBUF banks and bank size, the matmul tile cap, the
  weight tile-group sizes, the usable-datatype menu, the core count. These are the
  values `rocket-userspace`'s `rocket_hw_profile` collects.
- **The register offset map**: the numeric CNA/CORE/DPU register offsets. These are
  IP-**revision**-specific rather than shared across every Rockchip NPU. The RK3576 NPU re-packs
  the CNA block relative to the RK3588, in [rk3576.md](rk3576.md). The *block bases*, CNA at
  core+0x1000 and CORE at +0x3000, and the register *semantics* are shared. The within-block
  offsets are not.
- **SoC integration**: physical base addresses, clocks, resets, power domains,
  IRQs, IOMMU wiring. Pure device-map facts.

| SoC | NPU cores | Status | Sheet |
|---|---|---|---|
| RK3588 | 3 | Hardware-validated (the reference target) | [rk3588.md](rk3588.md) |
| RK3576 | 2 | Runs on hardware. int8 direct, depthwise and packed-image convolution, fp16 convolution, int8, fp16 and bf16 matmul, PPU pooling and the DPU LUT all compute bit-exactly; five ImageNet classifiers and two COCO detectors run at CPU-parity accuracy through `tflite-rocket` | [rk3576.md](rk3576.md), [rk3576-regcmd.md](rk3576-regcmd.md) |
| RK3566 | 1 | Planned (hardware incoming) | [rk3566.md](rk3566.md) |

A chip can need two sheets. The RK3576's parameter sheet covers identity, SoC integration and the
state of the port. Its register encoding lives separately in
[rk3576-regcmd.md](rk3576-regcmd.md), being large enough, and concrete enough as an implemented
emitter with a gate, to warrant its own file.

[porting-patterns.md](porting-patterns.md) is the cross-chip sheet. It states what the RK3588 and
RK3576 pair establishes about porting to a third part:

- Which classes of claim transfer as reliable priors, meaning the datapath algebra.
- Which are coin flips and must be re-measured, meaning anything downstream of a machine
  parameter.
- The six edit operations the register re-pack is built from.
- The method for matching a register by value and function rather than by offset.

Read it before assuming any RK3588 fact on a new SoC.

An IP-inherent fact that turns out to differ on a new chip is a bug in the claim, not
a per-SoC parameter. Promote the correction into the subsystem doc and note the SoC
that disproved the old scope. The RK3576 CNA re-pack did exactly that to the old
"register offsets are not SoC-specific" claim.
