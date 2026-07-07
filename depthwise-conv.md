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
channel), so the driver splits `C` into chunks of `Cc` channels — the largest multiple of
`G` that fits one pass — and runs each chunk as an independent single DW job. Concatenating
the chunks equals the whole (bit-exact; channels never interact), and each chunk is
structurally a smaller-`C` copy of an already-validated single DW job, so it carries no new
HW risk. `Cc == C` (the whole fits) collapses to one job. A single channel whose *own*
feature map is too large for one pass would need **spatial** tiling (materialised halo, like
the direct path) — not yet implemented; `rocket_conv2d_plan` returns `<0` for that case so
it falls back to CPU. `ROCKET_CONV_DW_DEBUG=1` prints the chosen `Cc` and chunk count.
Depthwise is consumed end-to-end by the `tflite-rocket` delegate (it accepts
`DEPTHWISE_CONV_2D`, reorders the TFLite `[1,KH,KW,C]` filter to the driver's `[C,KH,KW]`,
and a real int8 depthwise `.tflite` runs on the NPU through it).

## Register fields that differ from a direct convolution

Programming the mode bits above is **not sufficient** — a depthwise job that is
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
| `bs_ow_op` (`DPU_BS_OW_OP`) | `0` (BS bypassed) | **`0x80 − weight_zp`** (128) | `rkt_regcmd.c` |
| **weight channel group `G`** (host scatter) | n/a | **fp16 = 32** (int8 = 64) | HW sweep |

**The host weight-group `G` is the field with no Mesa-fp16 reference.** Mesa's int8
depthwise uses `G = WEIGHT_ATOMIC_SIZE·2 = 64` (= feature-atom 16 × 4). fp16 halves the
feature atom to 8, so the same 4× ratio gives **G = 32**. With the DPU registers correct,
G=64 gives `max_abs` 6–24 (channel-plausible-but-wrong) and **G=32 is bit-exact**. Sweep
`G` only with the DPU registers already correct: a G∈{32,64} sweep against the wrong
`size_e=1`/`surf_add=128` fails for *both* values and masks the layout answer.

- **`weights_kernels = 1`** is the depthwise tell — Mesa's "output_channels collapses
  to 1" branch. The kernel count is 1 because each group emits a single output channel;
  the `G` channels of a group are carried in the weight atom, not as kernels.

- **`size_e = 3` for depthwise**, even when the output is fp16. This *overrides* the
  natural fp16 rule (`size_e = bytes-1 = 1`) documented in
  [encodings/size-e-quirk.md](encodings/size-e-quirk.md). Depthwise forces the wider
  output-surface stride regardless of output byte width — another case of `size_e` not
  tracking the actual element size.

- **`surfaces_per_row` doubles for depthwise** (`SURF_ADD` = `2·OW·OH·2`). A direct conv
  with the same output dimensions uses half this value; copying the direct value leaves
  the DPU writing the output surface at the wrong stride.

- **`bs_ow_op = 0x80 − weight_zero_point`** (`DPU_BS_OW_OP`, so `128` for symmetric/
  zero-zp fp16 weights). This belongs to the DPU bias/zero-point (BS) stage; the
  validated direct fp16 path bypasses BS and leaves it `0`, but the depthwise job needs
  the `128`.

- **`feature_grains = 50 + stride_y + 1`** (`CNA_CONV_CON2`, ≈52) — the empirical
  constant Mesa comments as magic ("seems to pass the most tests"), *not* a size-derived
  value. Insensitive for direct convs (a derived `IH+1` works there [HW sweep]); baked
  into the depthwise path to match the reference.

## The mental model

Depthwise reuses the direct-conv datapath but reinterprets its output geometry: one
kernel per group (`weights_kernels = 1`), a wider write-out surface (`size_e = 3`,
`od_bypass = 0`, `surfaces_per_row ×2`, `bs_ow_op = 128`, `feature_grains = 52`). The MAC
array, feature/weight CBUF load, and pad/stride/dilation handling are identical to a
direct conv. **The registers are only half the job** — the host weight cube must use the
right channel group (`G = 32` for fp16, half the int8 group). Get any of these wrong and
the result is plausible per channel but spatially/channel-scrambled — the same failure
signature as setting only the `CONV_MODE`/`DW_EN` mode bits. The way to bring this up: make
the regcmd *provably* match a validated direct-conv emitter (only the intended depthwise
deltas differ), **then** sweep the one field with no reference (`G`).

## int8 depthwise — the int8-out on-chip-requant path

int8 DW uses an **int8-output writer with on-chip requant**, not the int32-raw + host
requant that fp16 DW uses. The "int32-raw" int8 DW writer (the int8 analog of fp16-out —
`size_e=7`, `surf_add ×8`, int32 output, host requant) HW-fails with a clean signature:
`got[2k] == ref[k]`, i.e. the MAC and weight group are correct but the int32 output lands
at **2× the within-plane stride**. Sweeping `size_e` / `surf_mult` / readback-C2 / `G` does
not fix it — it is a genuinely different writer mode. [HW sweep]

The Teflon capture (`teflon-dw-capture/`) shows why: **Mesa does int8 depthwise as
int8-output with on-chip requant, never int32-raw.** The `gen_conv2d_dw_int8` path (with
`conv_params_t.int8_out=1`) reproduces Mesa's int8-output writer bit-for-bit, validated by
`tests/replay_dw_mesa.c` (the regcmd over Mesa's captured BOs → output bit-exact vs
`mesa-output`, on degenerate and non-degenerate random input). The deltas from the
int32-raw writer:

- **`DPU_DATA_FORMAT = 0`** — int8 out / int8 in / int8 proc (`out_precision = int8`, not
  int32). The output is int8, so the int32-readback geometry never applies.
- **`CORE_MISC_CFG QD_EN = 1`** (the int8 *matmul* / int32-raw conv uses 0). Required for
  the requant writer.
- **`size_e = 3`, `SURF_ADD = dst_surf_stride·4`** — the int8-out stride (matches the
  capture's `SURFACE_ADD = 256 = OH·OW·4`), not the int32-raw `7 / ·8`. (Like fp16 DW,
  `size_e` does not track the actual output byte width.)
- **`CNA_PAD_CON1 = (input_zp & 0xff) − 0x80`** — border pad in the uint8-centered domain
  (`0x7e` for `in_zp = −2`). The float/int32 paths leave it `0`; new `npu_cna_desc.pad_con1`.
- **per-output-channel int32 bias added in the BS ALU**, fetched by BRDMA: `DPU_BS_CFG =
  BS_ALU_ALGO(2)|BS_ALU_SRC(1)|RELU/MUL bypass` (`0x20150`), `DPU_RDMA_BRDMA_CFG =
  BRDMA_DATA_USE(1)`, `DPU_RDMA_BS_BASE_ADDR = bias-cube IOVA`. The bias cube is
  `tflite_bias − Σ_kernel(w_uint8 − w_zp)·(in_zp − 0x80)` (Mesa's zero-point fold,
  `rkt_coefs.c`). New `npu_dpu_desc.bias_en`/`bias_base_addr`.
- **`OUT_CVT` requant** (new `npu_dpu_desc.out_cvt_offset`/`out_cvt_shift`, wired into the
  conv emitter where they were hardcoded 0): `offset = output_zp − 0x80`; `scale`/`shift`
  from the Mesa/QNNPACK float-bits formula
  ```
  conv_scale = in_scale·w_scale / out_scale;  bits = float_bits(conv_scale);
  shift = (126 − (bits>>23) + 16) − 1;   scale = ((bits>>9) & 0x7fff) + 1 | 0x4000;
  ```
  (verified it reproduces the capture's `SCALE=17675 / SHIFT=22 / OFFSET=0xffffff85`).
- **`bs_ow_op = 0x80 − weight_zp`** — same as fp16 DW.

**The correctness oracle for int8 DW is Teflon, not CPU TFLite.** Teflon's NPU int8 DW
diverges from the CPU int8 kernel by up to ~143 on full-range random int8 (its per-tensor
uint8-domain requant is its own approximation), so the bit-exact gate is capture-replay
(`replay_dw_mesa`), not a from-scratch TFLite reference. Teflon also forces per-tensor
quant; real per-channel DW filters need the BS_MUL per-OC multiply path (a follow-on) or
stay on the dequant→fp16-DW→requant boundary.

**Runtime + delegate.** The int8-out path is wrapped in a runtime `rocket_conv2d_dw_int8`
(host-packs the uint8-centered cubes + the zero-point bias fold, reads back `+0x80` to the
model domain; G=64; channel-tiled) and routed by the delegate for **per-tensor
symmetric-pad** DW under `--option native_int8=1`. The host domain constants are pinned
empirically against the capture (`tests/dw_dump_capture.py`: the weight + bias cubes
reproduce `mesa-weights`/`mesa-biases` byte-for-byte; correction = `Σ(w_u8−w_zp)·(in_zp−0x80)`).
Gate `tests/conv_dw_int8_runtime.c`: the runtime vs Teflon `mesa-output` = **0/4096
bit-exact** (raw filter+bias from the tflite model, independent); delegate
`max|delegate−Teflon| = 0`.
