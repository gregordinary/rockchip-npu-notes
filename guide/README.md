# Running your own compute on the RK3588 NPU

This is a start-to-finish guide to driving the Rockchip RK3588 NPU through the mainline
FOSS `rocket` DRM-accel driver. It ties together the components of this stack. Each step
links to the repository that owns the detail.

This guide assumes the component repositories are checked out alongside these notes:

```
rockchip-npu-notes/   # you are here: the hardware reference + this guide
rocket-userspace/     # librocketnpu: the userspace driver + matmul and op library
ggml-rocket/          # ggml backend: LLM / Whisper frontend
tflite-rocket/        # TFLite delegate: detection / Frigate frontend
ort-rocket/           # ONNX Runtime execution provider: vision and text encoders
rknpu-submit/         # optional submit provider for the vendor rknpu kernel driver
patches/rocket/       # optional kernel patches (clock, perf counters); patches repo, rocket/ scope
```

The component repositories live under `github.com/gregordinary/`, each by the name above.

## The stack

```
            rockchip-npu-notes        hardware reference (this repo)
                   │
   patches/rocket  │  rocket-userspace     userspace driver + matmul library
   (kernel side)   │      │
                   │   ┌──┴──────────┐
                   │ ggml-   tflite-  ort-     frontends (link the library)
                   │ rocket  rocket   rocket
```

`rocket` is a generic register-command submitter: the kernel runs whatever CNA→CORE→DPU
register program you hand it. The library expresses a matmul as a 1×1 convolution over
those blocks, and emits those programs. The frontends map a model's operators onto the
library. The optional patches tune the kernel side. For the hardware reasoning, start with
[../hardware-overview.md](../hardware-overview.md) and
[../matmul-as-conv.md](../matmul-as-conv.md).

## Prerequisites

- An RK3588 board running a mainline kernel with the `rocket` driver, exposing
  `/dev/accel/accel0`.
- An aarch64 build toolchain (on-device, or a cross toolchain).

## 1. Build the driver library

Build librocketnpu (the `rocket-userspace` repository) and run one of its matmul
tests against your device. This confirms the whole path works (the device shim, the
register-command generation, and the NPU itself) before any model is involved. The
library has no ML-framework dependencies and is usable on its own. See its README for
the exact build and test commands.

## 2. (Optional) Raise the NPU clock

The NPU compute clock boots pinned at 200 MHz, one-fifth of its 1 GHz spec max (one-third
of the 600 MHz operating point). The clock patch in `patches/rocket` raises it safely to a
600 MHz operating point (~1.43x prefill). The patch is a module rebuild that never touches
the kernel image or device tree. Apply it before benchmarking. The background is in
[../perf/clock.md](../perf/clock.md).

## 3. Choose a frontend

### ggml-rocket: LLMs and speech

ggml-rocket is a ggml backend built as a runtime-loadable `libggml-rocket.so`. It drops
into stock llama.cpp and whisper.cpp. Point `GGML_BACKEND_PATH` at the `.so`, and the
matmul operators offload to the NPU. Everything else falls back to the CPU.

It runs the Whisper encoder end to end and LLM prefill on the NPU. Decode stays on the
CPU, where the single-row GEMV is far more efficient. See its README for build and usage.

### tflite-rocket: detection

tflite-rocket is a TensorFlow Lite external delegate. TFLite loads it at runtime and
partitions the graph: supported ops run on the NPU, the rest on the CPU. The target
consumer is Frigate on mainline RK3588. See its README for build and integration.

### ort-rocket: ONNX encoders

ort-rocket is an ONNX Runtime execution provider. ONNX Runtime loads it for an unmodified
`.onnx` model. It offloads a recognized encoder (plain and windowed ViTs, ModernBERT text
encoders) and `ConvTranspose` nodes, and leaves the rest of the graph on the CPU. See its
README for build and usage.

### A vendor kernel

Most Rockchip distributions ship the vendor BSP kernel, where the NPU is an `rknpu` device and
not `/dev/accel/accel0`. `rknpu-submit` is a submit provider for that driver. Built against it,
`librocketnpu` and every frontend above run unchanged on a vendor kernel. See its README.

## 4. (Optional) Probe DMA registers

> **The RK3588 has no usable HW DMA byte counters.** The `0x2000` "amount"
> page hard-locks the SoC on read, and the legacy `0x8000` offsets are config-only. See
> [../perf/hw-byte-counters.md](../perf/hw-byte-counters.md).

So the perf-counter patch in `patches/rocket` ships only a safe, read-only DDMA register
probe (`0x8000`, disarmed unless loaded with `rocket_ddma_probe=1`). The probe
characterizes those registers and is not a bytes-moved counter. Infer DMA traffic from
wall-clock time, or measure it externally with a DDR/NOC PMU.

## Further reading

The reference notes in this repository explain why the stack behaves as it does:

- [../datatypes.md](../datatypes.md): the datatype capability matrix (int4, int8, int16, fp16, bf16, tf32)
- [../hardware-overview.md](../hardware-overview.md): the NPU at the level you need to drive it
- [../matmul-as-conv.md](../matmul-as-conv.md): how a matmul becomes a 1×1 convolution
- [../perf/not-mac-bound.md](../perf/not-mac-bound.md): why quantization saves memory, not time
- [../encodings/](../encodings/): precision fields, tile layouts, K-accumulation, and the hardware traps
