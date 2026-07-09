# Running your own compute on the RK3588 NPU

A start-to-finish guide to driving the Rockchip RK3588 NPU through the mainline FOSS
`rocket` DRM-accel driver. It ties together the
components of this stack; each step links to the repository that owns the detail.

This guide assumes the component repositories are checked out alongside these notes:

```
rockchip-npu-notes/   # you are here — the hardware reference + this guide
rocket-userspace/         # librocketnpu: the userspace driver + matmul library
ggml-rocket/        # ggml backend  — LLM / Whisper frontend
tflite-rocket/      # TFLite delegate — detection / Frigate frontend
patches/rocket/     # optional kernel patches (clock, perf counters); patches repo, rocket/ scope
```

The component repositories live at `github.com/gregordinary/{rocket-userspace, ggml-rocket, tflite-rocket, patches}`.

## The stack

```
            rockchip-npu-notes        hardware reference (this repo)
                   │
   patches/rocket  │  rocket-userspace     userspace driver + matmul library
   (kernel side)   │      │
                   │   ┌──┴──┐
                   │ ggml-   rocket-  frontends (link the library)
                   │ rocket  tflite
```

`rocket` is a generic register-command submitter: the kernel runs whatever CNA→CORE→DPU
register program you hand it. A matmul is expressed as a 1×1 convolution over those
blocks. The library emits those programs; the frontends map a model's operators onto
the library; the optional patches tune the kernel side. For the hardware reasoning
behind all of it, start with [../hardware-overview.md](../hardware-overview.md) and
[../matmul-as-conv.md](../matmul-as-conv.md).

## Prerequisites

- An RK3588 board running a mainline kernel with the `rocket` driver, exposing
  `/dev/accel/accel0`.
- An aarch64 build toolchain (on-device, or a cross toolchain).

## 1. Build the driver library

Build librocketnpu (the `rocket-userspace` repository) and run one of its matmul
tests against your device. This confirms the whole path works — the device shim, the
register-command generation, and the NPU itself — before any model is involved. The
library has no ML-framework dependencies and is usable on its own. See its README for
the exact build and test commands.

## 2. (Optional) Raise the NPU clock

The NPU compute clock boots pinned at 200 MHz — one-fifth of its 1 GHz spec max (one-third of the
600 MHz operating point). The
clock patch in `patches/rocket` raises
it safely to a 600 MHz operating point (~1.43× prefill), as a module rebuild that never
touches the kernel image or device tree. Apply it before benchmarking; the background is
in [../perf/clock.md](../perf/clock.md).

## 3. Choose a frontend

### LLMs and speech — ggml-rocket

ggml-rocket is a ggml backend built as a
runtime-loadable `libggml-rocket.so`. It drops into stock llama.cpp and whisper.cpp:
point `GGML_BACKEND_PATH` at the `.so` and the matmul operators offload to the NPU, with
everything else falling back to the CPU. It runs the
Whisper encoder end to end and LLM prefill on the NPU (decode stays on the CPU, where
the single-row GEMV is far more efficient). See its README for build and usage.

### Detection — tflite-rocket

tflite-rocket is a TensorFlow Lite
external delegate. TFLite loads it at runtime and partitions the graph: supported ops
run on the NPU, the rest on the CPU. The target consumer is Frigate on mainline RK3588.
See its README for build and integration.

## 4. (Optional) Probe DMA registers

> **There are no usable HW DMA byte counters on rk3588.** The real `0x2000` "amount"
> page hard-locks the SoC on read, and the legacy `0x8000` offsets are config-only. See
> [../perf/hw-byte-counters.md](../perf/hw-byte-counters.md).

The perf-counter patch in `patches/rocket` therefore ships only a safe, read-only
DDMA register probe (`0x8000`, disarmed unless loaded with `rocket_ddma_probe=1`) for
characterising those registers — not a bytes-moved counter. DMA traffic itself must
be inferred from wall-clock time, or measured externally with a DDR/NOC PMU.

## Further reading

The reference notes in this repository explain *why* the stack behaves as it does:

- [../datatypes.md](../datatypes.md) — the datatype capability matrix (int4/int8/int16/fp16/bf16/tf32)
- [../hardware-overview.md](../hardware-overview.md) — the NPU at the level you need to drive it
- [../matmul-as-conv.md](../matmul-as-conv.md) — how a matmul becomes a 1×1 convolution
- [../perf/not-mac-bound.md](../perf/not-mac-bound.md) — why quantization saves memory, not time
- [../encodings/](../encodings/) — precision fields, tile layouts, K-accumulation, and the hardware traps
