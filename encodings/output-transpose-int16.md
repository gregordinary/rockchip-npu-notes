# int16 has no native matmul output: `tp_org_en` + the transposed 8/16-bit writer

The int16 *conv* computes correct int16×int16 dot products (`precision=1`, see
[precision-field.md](precision-field.md)), but getting them **out of the DPU** has
no clean full-precision path. There are exactly two output regimes, and neither is
"full iteration + int32":

| regime | how | what you get |
|---|---|---|
| **no transpose** | default DPU output | int32, full-width, **saturates** to int32 range — but only **one output tile** is written (row 0, channels 1..16); iteration is broken regardless of M/N/qd_en/size_e/grains/kernel_groups (all swept, zero effect) |
| **`tp_org_en=1`** | DPU "original transpose" (DPU_BS_OW_CFG bit 27) | the **entire M×N buffer** is written, but as **8- or 16-bit** elements (`tp_precision`: 0=int8, 1=int16), **transposed**, and **saturating** to that width |

There is no register combination that gives full-iteration int32. This is unique to
int16 — int8 (→int32) and int4 (→int16) iterate fully on the plain path.

**No dtype but int16 lacks a full-iteration output.** Every other matmul iterates
fully (int8→int32, int4→int16, fp16→fp32); int16 alone has no full-iteration output
regime across the entire sweep, so it is not a native matmul *output* type.

## The DPU output-writer cluster [source-confirmed: Mesa `registers.xml`]

Four DPU fields control the transpose/output-width path (wired through
`gen_matmul_task`; **int16-only — fp16/int8/int4 regcmd is byte-identical**,
verified with `/tmp/diff16.c`):

| field | register | meaning |
|---|---|---|
| `mc_surf_out`  | DPU_DATA_FORMAT bit3   | how many surfaces serialize the DPU output |
| `tp_precision` | DPU_WDMA_SIZE_0 bit27  | **transpose precision: 0 = 8-bit, 1 = 16-bit** |
| `size_c_wdma`  | DPU_WDMA_SIZE_0 b26:16 | Size_c for the WDMA |
| `tp_org_en`    | DPU_BS_OW_CFG  bit27   | **enable original transpose** (unlocks full-buffer iteration) |

`tp_org_en=1` is the iteration unlock; `tp_precision=1` picks 16-bit elements (so a
small int16 result round-trips losslessly). `tp_precision` is a **single bit** (DPU_WDMA_SIZE_0
bit27): only `&1` matters. A sweep over byte-width-looking values (8/16/32/64/256) is all
even, so `&1==0` and 16-bit transpose is never enabled — the sweep reads as "no effect" while
testing nothing. Sweep `{0, 1}`, not byte widths.

## The transposed int16 output layout

With `tp_org_en=1, tp_precision=1` the output is **int16** at this element index
(0-based `m,n`; `na = n/4`). Strides were measured across M∈{4,8,16}, N∈{16,32,64}
(see `matmul_int16_rocket.c` PROBE mode) and **scale with M, not N** [HW sweep]:

```
slot(m,n) = 4·m  +  (na%4)  +  (na/4)·4M  +  (n%4)·16M
```

| component | stride |
|---|---|
| `m`          | 4    |
| `na%4`       | 1    |
| `na/4`       | 4·M  |
| `n%4` (lane) | 16·M |

**HW-verified bit-exact at N≤32** (`8×32×32`, `16×32×32`, `32×32×32`, `4×32×32` all
pass 100% vs the int16-saturated reference). At N≥64 the `n/16` super term stops
extrapolating linearly (elements n≥32 read wrong), so the native path is capped at
**per-task N≤32**. It is dense (== M·N slots) exactly at N=32.

Decode method: a layout-map probe (`ROCKET_INT16_PROBE=1`) feeds inputs making
each `C[m,n] = m·N + (n+1)` — a unique, small, decodable signature — then reads the
buffer as int8/int16/int32 simultaneously to identify both the element size (int16
under `tp_precision=1`) and the exact slot→(m,n) map. **Trap:** the lane stride is
`16·M`, but a probe run at `N=4M` cannot distinguish `16·M` from `N·4` (they coincide),
so a single-shape fit ambiguously reads `N·4`. Probe at least one shape with `N≠4M` to
pin the lane stride to `M`.

## Output saturation

The int16-output path **saturates** to int16 (and the broken-iteration int32 tile
saturates to int32 — 5 exact / 11 saturate / 0 wrap measured under near-full-range
inputs) [HW sweep]. NVDLA, which Mesa cites (`rkt_coefs.c:152`), notes a "48-bit CACC for INT16
… round and saturation … to 32-bit." int8/int4 never exposed an output-overflow
regime; int16 does, at whatever width the writer is in.

## Consequence

The native int16 path is a working **int16→int16 (saturating, N≤32)** HW primitive —
useful only with output requant, like a quantized conv layer. For a real
full-precision int16 matmul, decompose into int8: `rocket_matmul_int16_exact`
(4 int8 matmuls, int64 recombine — see [tile-layouts.md](tile-layouts.md)).
