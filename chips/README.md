# Per-SoC parameter sheets

| SoC | NPU cores | Status | Sheet |
|---|---|---|---|
| RK3588 | 3 | Hardware-validated. The reference target | [rk3588.md](rk3588.md) |
| RK3576 | 2 | Bit-exact on hardware on `rocket` 1.6.0: int8 direct, depthwise and packed-image convolution, fp16 convolution, int8, fp16 and bf16 matmul, PPU pooling, and the DPU LUT. Five ImageNet classifiers and two COCO detectors run at CPU-parity accuracy through `tflite-rocket` | [rk3576.md](rk3576.md), [rk3576-regcmd.md](rk3576-regcmd.md) |
| RK3566 | 1 | Not characterized | [rk3566.md](rk3566.md) |

Unless a note states otherwise, the notes outside this directory describe the RK3588.

## Common to every RKNPU

Most of what the other notes describe is common to every [RKNPU](../GLOSSARY.md), the
NVDLA-derived NPU on every Rockchip SoC that integrates it:

- The matmul-as-1×1-convolution model.
- The CNA→CORE→DPU sequence.
- The precision-field encodings.
- The BS/BN/EW/LUT datapath.
- The tile-layout algebra.
- The "no on-chip layout conversion" and "no hardware gather" facts.

These do not change from SoC to SoC, and each is stated once, in its subsystem note.

## Per-SoC differences

Each sheet in this directory covers what changes per SoC:

- **Machine parameters**: CBUF banks and bank size, the matmul tile cap, the
  weight tile-group sizes, the usable-datatype menu, the core count. These are the
  values `rocket-userspace`'s `rocket_hw_profile` collects.
- **The register offset map**: the numeric CNA/CORE/DPU register offsets. These are specific
  to the RKNPU **revision**. The RK3576 NPU re-packs the CNA block relative to the RK3588, in
  [rk3576.md](rk3576.md). The *block bases*, CNA at core+0x1000 and CORE at +0x3000, and the
  register *semantics* are common to every RKNPU. The offsets within a block are not.
- **SoC integration**: physical base addresses, clocks, resets, power domains,
  IRQs, IOMMU wiring. These are device-map facts.

The RK3576 has two sheets. The first, [rk3576.md](rk3576.md), gives the machine parameters,
SoC integration, the programming model, the errata, and the performance of the library entries
and networks. Its register encoding, implemented as an emitter with its own test, is in
[rk3576-regcmd.md](rk3576-regcmd.md). [rk3576-validation.md](rk3576-validation.md) records how
both sheets' facts were established. The llama.cpp W8A8 port built on the RK3576's int8 matmul is
in [../perf/rk3576-w8a8-llm.md](../perf/rk3576-w8a8-llm.md).

## Porting to another SoC

[porting-patterns.md](porting-patterns.md) states what the RK3588 and the RK3576 show about
porting to another SoC:

- Which classes of claim transfer as reliable priors, meaning the datapath algebra.
- Which are no better than chance and must be re-measured, meaning anything derived from a
  machine parameter.
- The six edit operations the register re-pack is built from.
- The method for matching a register by value and function rather than by offset.

Read it before relying on an RK3588 fact on another SoC.
