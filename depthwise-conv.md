# Depthwise convolution

A depthwise convolution is a **grouped** convolution where each input channel is
filtered independently into one output channel (`OC == IC`, one `KH×KW` kernel per
channel). On the RK3588 NPU it runs on the same CNA→CORE→DPU datapath as a direct
convolution ([matmul-as-conv.md](matmul-as-conv.md)), selected by:

- `CNA_CONV_CON1.CONV_MODE = 3`
- `CORE_MISC_CFG.DW_EN = 1`
- `DPU_FEATURE_MODE_CFG.CONV_MODE = 3` and `DPU_RDMA…FEATURE_MODE_CFG.CONV_MODE = 3`

The weight cube is packed per group rather than per output channel: `(C/G, KH, KW, G)`
where `G` is the channel group (the innermost weight atom). The host scatter must use
the same `G`. There is no on-chip reorder, same as every other op.

fp16 depthwise is bit-exact on the RK1 across every test shape (3×3 s1, 3×3 s2, 5×5) once
the fields below are right. [HW sweep]

**Tiling & integration.** A depthwise layer too large for one CBUF pass is **tiled over
channels**: each channel is filtered independently (`OC == IC`, one `KH×KW` kernel per
channel), so the driver splits `C` into chunks of `Cc` channels, the largest multiple of
`G` that fits one pass, and runs each chunk as an independent single DW job. Concatenating
the chunks equals the whole (bit-exact; channels never interact), and each chunk is
structurally a smaller-`C` copy of an already-validated single DW job, so it carries no new
HW risk. `Cc == C` (the whole fits) collapses to one job. A single channel whose *own*
feature map is too large for one pass would need **spatial** tiling (a materialized halo, like
the direct path), which is not implemented; `rocket_conv2d_plan` returns `<0` for that case so
it falls back to CPU. `ROCKET_CONV_DW_DEBUG=1` prints the chosen `Cc` and chunk count.
Depthwise is consumed end-to-end by the `tflite-rocket` delegate (it accepts
`DEPTHWISE_CONV_2D`, reorders the TFLite `[1,KH,KW,C]` filter to the driver's `[C,KH,KW]`,
and a real int8 depthwise `.tflite` runs on the NPU through it).

## Register fields that differ from a direct convolution

Programming the mode bits above is **not sufficient**: a depthwise job that is
otherwise a copy of a direct-conv job produces channel-plausible-but-wrong output. The
working reference (Mesa's `rocket` driver, which runs depthwise correctly) branches on
`depthwise` for these fields:

| Field (register) | direct conv | **depthwise** | source |
|---|---|---|---|
| `weights_kernels` (`CNA_WEIGHT_SIZE2.WEIGHT_KERNELS`) | `align(OC, 2)` | **1** | `rkt_task.c` |
| `size_e` (`DPU_BS_OW_CFG.SIZE_E_{0,1,2}`) | `1` (fp16 out) | **3** | `rkt_regcmd.c` |
| `od_bypass` (`DPU_BS_OW_CFG.OD_BYPASS`) | `1` | **0** (unset) | `rkt_regcmd.c` |
| `surfaces_per_row` (`DPU_SURFACE_ADD.SURF_ADD`) | `OW·OH·2` | **`OW·OH·2 · 2`** | `rkt_task.c` |
| `feature_grains` (`CNA_CONV_CON2`) | insensitive | **`50+stride_y+1`** | `rkt_regcmd.c` |
| `bs_ow_op` (`DPU_BS_OW_OP`) | `0` (BS bypassed) | **`0x80 − weight_zp`**: 128 at fp16, 0 for int8 (below) | `rkt_regcmd.c` |
| **weight channel group `G`** (host scatter) | n/a | **fp16 = 32** (int8 = 64) | HW sweep |

**The host weight-group `G` is the field with no Mesa-fp16 reference.** Mesa's int8
depthwise uses `G = WEIGHT_ATOMIC_SIZE·2 = 64` (= feature-atom 16 × 4). fp16 halves the
feature atom to 8, so the same 4x ratio gives **G = 32**. With the DPU registers correct,
G=64 gives `max_abs` 6-24 (channel-plausible-but-wrong) and **G=32 is bit-exact**. Sweep
`G` only with the DPU registers already correct: a G∈{32,64} sweep against the wrong
`size_e=1`/`surf_add=128` fails for *both* values and masks the layout answer.

- **`weights_kernels = 1`** is the depthwise tell, from Mesa's "output_channels collapses
  to 1" branch. The kernel count is 1 because each group emits a single output channel;
  the `G` channels of a group are carried in the weight atom, not as kernels.

- **`size_e = 3` for depthwise**, even when the output is fp16. This *overrides* the
  natural fp16 rule (`size_e = bytes-1 = 1`) documented in
  [encodings/size-e-quirk.md](encodings/size-e-quirk.md). Depthwise forces the wider
  output-surface stride regardless of output byte width, another case of `size_e` not
  tracking the actual element size.

- **`surfaces_per_row` doubles for depthwise** (`SURF_ADD` = `2·OW·OH·2`). A direct conv
  with the same output dimensions uses half this value; copying the direct value leaves
  the DPU writing the output surface at the wrong stride.

- **`bs_ow_op = 0x80 − weight_zero_point`** (`DPU_BS_OW_OP`, so `128` for symmetric/
  zero-zp fp16 weights). This belongs to the DPU bias/zero-point (BS) stage; the
  validated direct fp16 path bypasses BS and leaves it `0`, but the depthwise job needs
  the `128`.

- **`feature_grains = 50 + stride_y + 1`** (`CNA_CONV_CON2`, ~52), the empirical
  constant Mesa comments as magic ("seems to pass the most tests"), *not* a size-derived
  value. Insensitive for direct convs (a derived `IH+1` works there [HW sweep]); baked
  into the depthwise path to match the reference.

## The mental model

Depthwise reuses the direct-conv datapath but reinterprets its output geometry: one
kernel per group (`weights_kernels = 1`), a wider write-out surface (`size_e = 3`,
`od_bypass = 0`, `surfaces_per_row ×2`, `bs_ow_op`, `feature_grains = 52`). The MAC
array, feature/weight CBUF load, and pad/stride/dilation handling are identical to a
direct conv. **The registers are only half the job**: the host weight cube must use the
right channel group (`G = 32` for fp16, half the int8 group). Get any of these wrong and
the result is plausible per channel but spatially/channel-scrambled, the same failure
signature as setting only the `CONV_MODE`/`DW_EN` mode bits. The way to bring this up: make
the regcmd *provably* match a validated direct-conv emitter (only the intended depthwise
deltas differ), **then** sweep the one field with no reference (`G`).

## int8 depthwise: the int8-out on-chip-requant path

int8 DW uses an **int8-output writer with on-chip requant**, not the int32-raw + host
requant that fp16 DW uses. The int32-raw int8 DW writer (`size_e=7`, `surf_add ×8`, int32
output) fails with a clean signature: `got[2k] == ref[k]`. The MAC and the weight group are
correct, and the int32 output lands at **2x the within-plane stride**. Sweeping `size_e`,
`surf_mult`, the readback C2 and `G` does not fix it, because it is a different writer mode.
[HW sweep]

The int8-output writer is Mesa's (`rkt_regcmd.c`). `gen_conv2d_dw_int8` in rocket-userspace
emits it with `conv_params_t.int8_out=1`. Its deltas from the int32-raw writer:

- **`DPU_DATA_FORMAT = 0`**: int8 out, in and proc (`out_precision = int8`). The int32-readback geometry never applies.
- **`CORE_MISC_CFG QD_EN = 1`**, which the requant writer requires. The int8 matmul and the int32-raw conv use 0.
- **`size_e = 3`, `SURF_ADD = dst_surf_stride·4`**: the int8-out stride, the capture's `SURFACE_ADD = 256 = OH·OW·4`. Like fp16 DW, `size_e` does not track the output byte width.
- **A per-output-channel int32 bias in the BS ALU**, fetched by BRDMA: `DPU_BS_CFG` `0x20150`, `DPU_RDMA_BRDMA_CFG = BRDMA_DATA_USE(1)` and `DPU_RDMA_BS_BASE_ADDR` set to the bias cube.
- **The `OUT_CVT` requant**, whose scale and shift come from the formula below and whose offset is the output zero point. [encodings/out-cvt-converter.md](encodings/out-cvt-converter.md) owns the converter.
- **`CNA_PAD_CON1`** and **`DPU_BS_OW_OP`**, each with a trap of its own below.

```
conv_scale = in_scale·w_scale / out_scale;  bits = float_bits(conv_scale);
shift = (126 − (bits>>23) + 16) − 1;   scale = ((bits>>9) & 0x7fff) + 1 | 0x4000;
```

The formula reproduces a Teflon capture's `SCALE=17675` and `SHIFT=22`. [source-confirmed]

### The int8 domain

Mesa's rocket driver is a uint8 driver: its formulas take a uint8 tensor `u` with zero
point `Z` [source-confirmed, `rkt_ml.c`, `rkt_coefs.c`, `rkt_regcmd.c`]. An int8 tensor is
the uint8 tensor `u = x + 0x80` with `Z = zp + 0x80`, at the same scale. Substituting gives
the int8 program:

| Field | Mesa's uint8 formula | int8 value |
|---|---|---|
| Input and weight cubes | `u − 0x80` | the raw int8 values |
| `CNA_PAD_CON1` | `Z_in − 0x80` | `in_zp`, one byte per lane |
| `OUT_CVT` offset | `Z_out − 0x80` | `out_zp` |
| `DPU_BS_OW_OP` | `0x80 − Z_w` | 0 for symmetric weights |
| Bias | `bias − (Z_in − 0x80)·Σ(u_w − Z_w)` | `bias − in_zp·Σ_kernel w` |
| Output | `npu_byte + 0x80` | the raw int8 byte |

This program computes TFLite's int8 depthwise [HW sweep, RK1, 600 MHz, measured
2026-09-23]. Against TFLite's reference kernels on one model layer with a random input, 11
of 4096 outputs differ by one and none by more. The difference is the requant: a 15-bit
multiplier against TFLite's 31-bit one moves an output across a rounding boundary.
Against a host model of TFLite's int32 accumulator followed by the `OUT_CVT` requant, the
program is bit-exact at these shapes:

| Channels × plane | Kernel, stride | Zero points (in, out) |
|---|---|---|
| 64×8×8 | 3×3, 1×1 | -2, 5 |
| 64×6×10 | 3×3, 1×1 | 37, -20 |
| 64×9×7 | 3×3, 2×1 | -128, 127 |
| 128×7×5 | 3×3, 1×2 | 127, -128 |
| 64×8×11 | 5×5, 1×1 | 0, 0 |

The weights are symmetric throughout. A nonzero weight zero point adds `w_zp·Σx`, which
depends on the input and cannot fold into the bias. rocket-userspace refuses it.

### `CNA_PAD_CON1` is read per byte lane

On the int8 depthwise path, the odd channels among the first 32 of a 64-channel group read
byte 1 of `CNA_PAD_CON1`, and every other channel reads byte 0 [HW sweep, RK1]. The pad byte
therefore goes in every lane. A single byte sign-extended to 32 bits pads those 16 lanes with
`0x00` or `0xFF`, which only the border outputs of those channels show. Mesa's general
formula writes exactly that, and its two hand-coded values, `0xffff8080` and `0x0b0b`, are
the byte written twice [source-confirmed, `rkt_regcmd.c`].

The method reads the pad directly. One unit weight sits at kernel tap (0,0), with bias 0,
unit scales and the input equal to `in_zp` everywhere, so each channel's corner output is
`pad − in_zp`. At `in_zp` 37, -2, 100 and -100 over 64 channels at 6×10, those 16 lanes read
byte 1 and the other 48 read byte 0. Whether the pattern repeats per 64-channel group was
not isolated. With the byte in every lane the question does not arise, and the 128-channel
shape above is bit-exact.

### `DPU_BS_OW_OP` is the CPEND operand

The TRM names `DPU_BS_OW_OP` the CPEND operand, and `DPU_BS_OW_CFG.OD_BYPASS` bypasses CPEND
[TRM]. On the int8 depthwise path it must be 0 for symmetric weights, Mesa's `0x80 − Z_w` at
the uint8-equivalent `Z_w = 128`. At 128, 3962 of 4096 outputs of a real layer are wrong
[HW sweep, RK1]. That is the value the same formula gives when an int8 weight zero point of
0 is read as a uint8 one. What CPEND computes was not isolated: a term in the weight zero
point is the likely reading, since Mesa's formula carries one [expected]. The fp16 depthwise
path runs bit-exact at 128.

### Captures as oracles

A capture from a reference driver is an oracle only for what that driver supports. The one
int8 depthwise capture ran an int8 model through Mesa's uint8 formulas, so it records those
formulas applied to int8 bytes. On an int8 byte, `u − 0x80` is the flip `x ^ 0x80`, which is
not linear in `x`. Every one of the capture's 4096 outputs differs from TFLite's for the same
model and input, by up to 133 [HW sweep, LiteRT 2.1.6 reference kernels].

The capture's input is also constant zero, so it cannot see the input-cube addressing either.
Replaying a register program over its buffers says the program is Mesa's, and says nothing
about the function it computes. The oracle for the function is TFLite.

### Runtime and delegate

`rocket_conv2d_dw_int8` in rocket-userspace wraps the path. It packs the raw cubes and the
bias fold, reads the output back raw, uses `G = 64` and tiles over channels. It takes
per-tensor quant with symmetric weights. The `tflite-rocket` delegate routes a per-tensor,
symmetric-pad int8 depthwise to it under `--option native_int8=1`. A per-channel filter
needs the BS_MUL per-channel multiply and stays on the fp16 depthwise path.
