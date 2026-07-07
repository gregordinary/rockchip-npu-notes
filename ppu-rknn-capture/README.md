# Vendor PPU pooling capture (rknn-toolkit2 → .rknn → decode)

First-party ground truth for on-NPU pooling. The capture: build a set of 1-op ONNX pool
models, convert each to a `.rknn` for `rk3588` with `rknn-toolkit2` (offline compile — no
NPU needed; the rk3588-target regcmd is identical on any host), then walk the `.rknn` for
the 64-bit NPU command words and decode the PPU (`0x6000`) / PPU_RDMA (`0x7000`) page. The
gahingwoo method ([SOURCES.md](../SOURCES.md)).

It gives the exact average-pool `RECIP_KERNEL` fixed-point format directly, rather than a
host approximation. The `.rknn` and decoded transcripts are build artifacts — regenerate
them with the script below; they are not committed.

## Repro

```sh
/tmp/rknnvenv/bin/python gen_pool_rknn.py        # build ONNX + convert to .rknn
/tmp/rknnvenv/bin/python decode_pool_rknn.py *.rknn
```

venv: `python3 -m venv --without-pip /tmp/rknnvenv`; `get-pip.py`; then
`pip install rknn-toolkit2 "setuptools<81" "onnx==1.16.0"` (setuptools 82 dropped
`pkg_resources`; onnx ≥1.17 dropped `onnx.mapping` — rknn-toolkit2 2.3.2 needs both).

## The decoded PPU pooling program (vendor, byte-level; max-pool 2×2 s2, in 1×8×4×4)

Each is one `NPUOP(target,value,reg)`. Geometry fields are **(value − 1)**; strides are
**bytes** (regs hold a [31:4] field = bytes, 16-aligned). C2 = 8 (fp16 cube).

| reg | field(s) | meaning |
|---|---|---|
| `PPU_S_POINTER` 0x6004 | 0xE | POINTER_PP_MODE\|EXECUTER_PP_EN\|POINTER_PP_EN (== DPU) |
| `PPU_RDMA_S_POINTER` 0x7004 | 0xE | same |
| `…_CUBE_IN_WIDTH/HEIGHT/CHANNEL` | iw-1, ih-1, c-1 | input cube dims |
| `…_CUBE_OUT_WIDTH/HEIGHT/CHANNEL` | ow-1, oh-1, c-1 | output cube dims |
| `PPU_OPERATION_MODE_CFG` 0x6024 | FLYING_MODE(1)\|POOLING_METHOD | **max → METHOD=1 (0x11), avg → METHOD=0 (0x10)** |
| `PPU_POOLING_KERNEL_CFG` 0x6034 | KW-1\|(KH-1)<<8\|(SX-1)<<16\|(SY-1)<<20 | kernel + stride, all −1 |
| `PPU_RECIP_KERNEL_WIDTH/HEIGHT` 0x6038/3C | **fp16(65536/k)** | avg only; 0 for max |
| `PPU_POOLING_PADDING_CFG` 0x6040 | L\|T<<4\|R<<8\|B<<12 | pad counts (NOT −1) |
| `PPU_PADDING_VALUE_1/2_CFG` | 0 | pad fill |
| `PPU_DST_BASE_ADDR` 0x6070 | output IOVA | runtime-patched |
| `PPU_DST_SURF_STRIDE` 0x607C | oh·ow·C2·2 bytes | output per-C-plane stride |
| `PPU_DATA_FORMAT` 0x6084 | INDEX_ADD(=dst_surf/16)\|PROC_PRECISION(2) | = dst_surf_bytes \| 2 (fp16) |
| `PPU_MISC_CTRL` 0x60DC | BURST_LEN(3) | |
| `PPU_RDMA_CUBE_IN_*` | iw-1, ih-1, c-1 | RDMA input dims |
| `PPU_RDMA_SRC_BASE_ADDR` 0x701C | input IOVA | runtime-patched |
| `PPU_RDMA_SRC_LINE_STRIDE` 0x7024 | iw·C2·2 bytes | input row stride |
| `PPU_RDMA_SRC_SURF_STRIDE` 0x7028 | ih·iw·C2·2 bytes | input per-C-plane stride |
| `PPU_RDMA_DATA_FORMAT` 0x7030 | IN_PRECISION(2) | fp16 |
| PC trailer | `PC_OPERATION_ENABLE` (target 0x81) = **0x60** | PPU_OP_EN(b5)\|PPU_RDMA_OP_EN(b6) |

**No per-block `OPERATION_ENABLE`** (0x6008/0x7008) is emitted — the PC `0x60` mask alone
brings PPU + PPU_RDMA into op_en (a per-block OP_EN also works; the minimal form is `0x60`
only). The enable value `0x60` carries **no bit0**
(bit0 = CNA_OP_EN, unused here) — confirming PC_OPERATION_ENABLE is the global
block-participation mask (matmul = 0x1D = CNA|CORE|DPU|DPU_RDMA), not a "task start" bit.

## Flying vs standalone

`OPERATION_MODE_CFG.FLYING_MODE = 1` **and** PPU_RDMA is fully armed (own SRC_BASE +
strides + OP via the PC mask), while `PPU_DATA_FORMAT.DPU_FLYIN = 0`. So a standalone pool
is a **self-contained PPU job fed by PPU_RDMA**; FLYING_MODE=1 is set regardless (it is not
"epilogue-of-conv" here — that would be DPU_FLYIN=1).

## RECIP_KERNEL format

`recip_axis = fp16(65536.0 / k_axis)`, stored in the 17-bit field as the fp16 bit pattern.
The PPU computes `avg = sum · recip_w · recip_h · 2⁻³²` = `sum/(kw·kh)`.

| k | observed | fp16(65536/k) |
|---|---|---|
| 2 | `0x7800` | 32768.0 → `0x7800` OK |
| 3 | `0x7555` | 21845.3 → `0x7555` OK |
| 2×3 (asymmetric) | h=`0x7800`, w=`0x7555` | per-axis OK |

A host approximation that hard-codes `30720` (= `0x7800` = fp16(2¹⁶/2)) for **all** k and
post-scales is correct only at k=2. The per-axis `fp16(65536/k)` format above is exact and
needs no host correction.
Precision: fp16(65536/k) has ~3-4 sig figs, so avg carries ~0.02-0.05% recip-quant error
(matches the vendor exactly — same recip). k must be ≥ 2 (k=1 → 65536 overflows fp16).

Note: the **k=4 global** `.rknn` (`avgpool_4x4s1_c8`, `gavgpool_c16`) was lowered by
rknn-toolkit2 into many small ops (not a single clean PPU pool), so its RECIP isn't a clean
read — the k=2 + k=3 + 2×3 captures are conclusive on their own.
