# Teflon EW-operand capture

Capture for the **EW two-operand RE** ([../encodings/dpu-lut-activation.md](../encodings/dpu-lut-activation.md)):
how the rocket DPU reads a *second*
tensor for an elementwise op (the question that blocks a fully-on-NPU HardSwish/SiLU
`x·gate(x)` multiply).

## Files
- `add_c8_4x4.tflite` — minimal int8 per-tensor model: `add(conv_a(x), conv_b(x))`
  (two 1×1 convs added; both add-inputs are conv outputs, the only shape Teflon's
  fuser accepts — a MobileNet residual `add(input, conv)` **asserts** `input_op_2`
  in `rkt_ml.c:344` because the raw graph input has no producing op).
- `regcmd.decoded.txt` — `decode.py` of the captured `mesa-regcmd-000-000.bin`.

Reproduce on the RK1:
```
~/tfvenv/bin/python3 .../tools/make_add_tflite.py --c 8 --hw 4
ROCKET_DEBUG=dbg_msgs,dump_bos ~/tfvenv/bin/python3 .../tools/run_delegate.py \
  add_c8_4x4.tflite --delegate .../libteflon.so
python3 .../rocket/decode.py --xml .../rocket/registers.xml --dump mesa-regcmd-000-000.bin
```

## What the decode shows (the baseline)

The dumped task is a **plain conv** — Teflon's TFLite partitioner only claimed one
conv for this graph, so the EW residual didn't appear in the dump. But the plain
conv is itself the key baseline: it confirms the **main feed is the conv/CACC, with
MRDMA disabled**:

```
DPU_EW_CFG            = EW_*_BYPASS(1) ...            # EW fully bypassed
DPU_RDMA_ERDMA_CFG   = ERDMA_DISABLE(1)              # no operand feed
DPU_RDMA_FEATURE_MODE_CFG = BURST_LEN(15) | MRDMA_DISABLE(1)   # MRDMA OFF
```

## The operand recipe (from Mesa `rkt_regcmd.c`, the authoritative source)

For `operation->add_tensor != -1` (a conv with an EW residual), the second operand
is delivered as:

```
DPU_EW_CFG          = EW_CVT_TYPE(1)|EW_DATA_MODE(1)|EDATA_SIZE(1)|EW_ALU_ALGO(2=add)
                      |EW_RELU_BYPASS(1)|EW_LUT_BYPASS(1)|EW_OP_SRC(1)
DPU_RDMA_SRC_BASE_ADDR = add_tensor              # MRDMA SRC = the operand
DPU_RDMA_ERDMA_CFG     = ERDMA_DATA_MODE(1)|ERDMA_DATA_SIZE(1)
DPU_RDMA_EW_BASE_ADDR  = add_tensor + offset     # ERDMA EW = the operand
DPU_RDMA_EW_SURF_STRIDE = ew_stride
DPU_RDMA_FEATURE_MODE_CFG = BURST_LEN(15)|COMB_USE(5)   # MRDMA *enabled*, combined
DPU_RDMA_SURF_NOTCH / EW_SURF_NOTCH = surf_notch
```

**The conclusion:** in *both* the plain-conv and the residual-add cases the DPU
**main feed is the conv/CACC**. MRDMA is either OFF (plain) or repurposed — with
`COMB_USE(5)` — to deliver the *operand* (SRC_BASE and EW_BASE both point at the
add tensor). MRDMA is never simultaneously a flying main AND an operand feed.

So the flying-mode LUT activation (MRDMA flying = main, no operand) and a two-buffer
EW op are different MRDMA roles. A pure flying-MRDMA-main + ERDMA-operand multiply
(our `gen_ew_mul_fp16`) has no valid main once MRDMA is needed for the operand →
the operand reads 0. **A fully-on-NPU two-buffer EW multiply requires a conv (even
an identity 1×1) as the main feed.**

Feeding the EW operand path with an **identity matmul** as the main (the fp16 K-accum
machinery, `gen_matmul_fp16` `accumulate=1`) and switching the EW op add→mul
(`DPU_EW_CFG` `0x108202C0` → **`0x108003C4`**, i.e. `EW_OP_TYPE(1)`) computes `A*B`
**bit-exact** on the NPU [HW sweep] (`tests/ew_mul_rocket.c`; ADD reproduces `A+B` on the
same datapath). This is wired into `rocket_ew_mul_fp16` and the fully-on-NPU
HardSwish/SiLU (`ROCKET_ACT_NPU_MUL=1`). The operand transport here (ERDMA `EW_BASE` +
MRDMA `SRC_BASE` + `COMB_USE(5)`) is the add-path's verbatim — only the EW op field
differs. Detail: [../encodings/dpu-lut-activation.md](../encodings/dpu-lut-activation.md).
