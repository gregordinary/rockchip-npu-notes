# Vendor DPU LUT capture (rknn-toolkit2 -> .rknn -> decode)

This capture reads the DPU LUT activation program the vendor compiler emits for the RK3588. It
builds one-op ONNX Sigmoid and Tanh models. `rknn-toolkit2` compiles each to a `.rknn` for
`rk3588`, offline and with no NPU. The decoder reads the `.rknn`'s register commands and its
LUT table upload. The `.rknn` files and the decoded transcripts are build
artifacts and are not committed.

## Repro

```sh
<venv>/bin/python gen_lut_rknn.py OUTDIR
python3 decode_lut_rknn.py OUTDIR/sigmoid_c16_f16.rknn OUTDIR/tanh_c16_f16.rknn
python3 decode_lut_rknn.py --table OUTDIR/tanh_c16_f16.rknn
```

The venv is the one [../ppu-rknn-capture/README.md](../ppu-rknn-capture/README.md) builds.
The results below were compiled with `rknn-toolkit2` 2.3.0. Two compiles of the same model
decode identically.

## The compiled program

Each model compiles to two tasks. The first writes the whole DPU register block and uploads
the table. Its register block holds one `LUT_ACCESS_DATA` write. The 513 LE entries follow
`LUT_ACCESS_CFG` `0x00020000`, and the 513 LO entries follow `0x00030000`. The second is the
activation itself, a DPU flying-mode task with `S_POINTER` `0x30`. The two operators differ in
these words of the second task:

| Register | Sigmoid | Tanh |
|---|---|---|
| `BN_MUL_CFG` `0x4068` | `0x69120000`, fp16(2596) | `0x6d4c0000`, fp16(5424) |
| `BN_ALU_CFG` `0x4064` | `0x80000000` | `0x80000000` |
| `OUT_CVT_OFFSET` `0x4080` | `1` | `1` |
| `OUT_CVT_SCALE` `0x4084` | `0x00010001`, `FP32TOFP16_EN` and 1 | `0x00010001` |
| `OUT_CVT_SHIFT` `0x4088` | `0x0000f000`, `MINUS_EXP` 15, `CVT_TYPE` 0 | `0x0000f000` |
| `LUT_INFO` `0x410C` | `0x00050500`, index select 5 and 5 | `0x00050500` |
| `LUT_LE_START` / `LUT_LO_END` | `0xffffc000` / `0x00004000` | `0xffffc000` / `0x00004000` |
| `LUT_LE_SLOPE_SCALE` / `_SHIFT` | `0x5a43` (23107) / `0x16` (22) | `0` / `0` |
| Table | Unsigned Q0.15, 59 to 32708 | Signed Q0.15, -32611 to 32611 |

[source-confirmed: `rknn-toolkit2` 2.3.0, `rk3588`, fp16]

Sigmoid's geometry is the one `librocketnpu` programs
([../encodings/dpu-lut-activation.md](../encodings/dpu-lut-activation.md) §"The geometry
constants (sigmoid)"), apart from `CVT_TYPE`, which the library sets to 1. At `OUT_CVT_SCALE` 1
the two orders of the offset and the multiply give the same value. The library computes its own
table from the function, and it differs from the vendor's in 787 of 1026 entries, by at most 3
LSB.

The vendor's tanh is a different program from the library's. It stores tanh itself as a signed
Q0.15 table over x in ±3.02 and decodes it with the sigmoid's converter. The library stores
`(tanh(x)+1)/2` unsigned over x in ±4 and decodes it with the affine `OUT_CVT`. That form is in
[../encodings/dpu-lut-activation.md](../encodings/dpu-lut-activation.md) §"Signed / wide output:
the affine OUT_CVT".
