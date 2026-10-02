# int16 matmul output: the int32 writer and the transposed writer

An int16 × int16 matmul contracts at precision 1 and has two output writers. The plain
DPU path writes a full int32 surface in the standard `[N/4,M,4]` output cube, saturating
to int32. The transposed path (`tp_org_en`) writes 8- or 16-bit elements in a layout of
its own, truncated rather than saturated. Each needs its own output geometry, and each
hangs under the other's. Scope: one task, single K pass, the RK3588.

| writer | registers | what you get |
|---|---|---|
| int32 | `size_e` 3, `surf_add` = stride × 4, `out_precision` 4 | the whole M×N surface as int32, **saturating**, exact against the int64 dot product clamped to int32 |
| transposed | `tp_org_en`=1 (DPU_BS_OW_CFG bit 27), `tp_precision` 0 or 1, `size_e` 7, surface add × 8 | the whole M×N surface as int8 or int16, transposed. The int16 form holds the **low 16 bits** of the sum clamped to int32, verified at N<=32 |

## The int32 writer

The int32 writer takes the output geometry of fp16 -> fp32, which also reads two-byte
operands and writes four-byte results. That is `size_e` 3 and a surface add of the
destination stride times 4. The CNA and CORE side is the fp16 matmul's, with the precision
fields at 1 and `qd_en` at 1.

The int32 writer is bit-exact on every element at six shapes with M, K and N all
different (M 4-100, K 32-1024, N 48-256). The reference is the int64 dot product saturated
to int32. The run covers four operand fills at two reps each, 48 of 48 arms [HW sweep,
RK3588, `rocket` 1.3.0, 600 MHz, 2026-09-26, `tests/int16_native_probe`]:

| fill | A | B | what it checks |
|---|---|---|---|
| small | [-16, 15] | [-16, 15] | the range earlier evidence used, whose high byte is a sign extension |
| afull | all of int16 | within the int32 bound for K | A's high byte |
| bfull | within the int32 bound for K | all of int16 | B's high byte |
| big | all of int16 | all of int16 | sums past int32, scored against the saturated model |

An independent generator agrees on the same silicon under the vendor `rknpu` driver.
allbilly's `experimental/gemm_int16.py` writes the same `size_e` 3 pair. It is exact under
all four fills at those six shapes and at 64³ and 256³ [HW sweep, vendor kernel
`6.1.172-vendor-rk35xx`, `rknpu` 0.9.8, 2026-09-26]. Its input layout differs from this
project's at non-square shapes, so the agreement covers the writer rather than one program.

### The accumulator and the saturation

The `big` fill reaches dot products of 2^35.5 at K=1024. Every result that lands back
inside int32 had a running sum that left it on the way, and all of them are exact. Every
result outside int32 reads as the clamped value, not the wrapped one. So the accumulator
holds at least 36 bits, and the writer saturates on the way out [HW sweep]. NVDLA's note
that Mesa cites (`rkt_coefs.c:152`) gives a 48-bit CACC for INT16 that rounds and
saturates to 32 bits [source-confirmed]. That agrees, and nothing here reached past 2^35.5.

Saturation sets what the int32 writer is exact for. A product of two full-range int16
values is up to 2^30, so two of them can pass int32. When the operand ranges keep
`|sum| < 2^31` over the task's K, a result is exact. One operand full-range and the other
within `(2^31-1) / (K · 32768)` is one such pair. For full-range operands at any useful K,
the int64-exact route is the int8 byte decomposition below.

### int8 output geometry on an int16 task

**Do not give an int16 program the int8 integer-output geometry**, `size_e` 7 with the
surface add × 8 ([size-e-quirk.md](size-e-quirk.md)). The DPU writes row 0's first sixteen
channels, exact, and the task never completes. `rocket` retires it at its 500 ms watchdog
and signals the fence as if it had finished. So `PREP_BO` returns 0, and the buffer reads
as one correct 1×16 tile over whatever it held before. All 48 of 48 such tasks did this,
and the kernel logged each as a job timeout [HW sweep].

A surface that reads "one tile, the rest untouched" is a retired task, not a missing
output mode. Read the kernel log's timeout count, or the wait time, before reading the
surface.

## The transposed writer

With `tp_org_en=1` the DPU writes the whole M×N buffer as 8- or 16-bit elements,
transposed. `tp_precision` (DPU_WDMA_SIZE_0 bit 27) picks the width: 0 is 8-bit and 1 is
16-bit. It is a single bit. A sweep over byte-width-looking values (8, 16, 32, 64, 256)
writes only even values and never sets it, so it reads as no effect while testing nothing.
Sweep `{0, 1}`.

**The 16-bit form truncates.** Each element is the low 16 bits of the dot product after
the sum is clamped to int32. A sum past int16 wraps, and a sum past int32 reads -1 above
and 0 below. That model is exact on every element at 15 arms [HW sweep, RK3588, `rocket`
1.3.0, 2026-09-26, `matmul_int16_rocket` with `ROCKET_INT16_TP16=1`]. The arms span M
4-32, K 32-64, N 16-32, with operands in ±16, ±1024 and all of int16.

None of the elements outside int16 equaled a clamp to int16. The 8-bit form was not
scored. So this writer is exact only where the sums fit int16, and a requantizing caller
must bound them.

The transposed writer's geometry is int8's, `size_e` 7 with the surface add × 8, and all
15 arms are exact there. Under the int32 writer's `size_e` 3 it writes half the surface or
less, and at M >= 16 the task hangs to the watchdog.

**At N=16 the task hangs after a complete write.** `12×64×16` writes every element right,
and the 500 ms watchdog still retires it. That held on 4 of 4 runs, at three
ranges [HW sweep]. A correct surface does not show that the task completed.

Four DPU fields control this path [source-confirmed: Mesa `registers.xml`]:

| field | register | meaning |
|---|---|---|
| `mc_surf_out`  | DPU_DATA_FORMAT bit 3   | how many surfaces serialize the DPU output |
| `tp_precision` | DPU_WDMA_SIZE_0 bit 27  | transpose precision: 0 = 8-bit, 1 = 16-bit |
| `size_c_wdma`  | DPU_WDMA_SIZE_0 b26:16  | Size_c for the WDMA |
| `tp_org_en`    | DPU_BS_OW_CFG bit 27    | enable original transpose |

With `tp_org_en=1, tp_precision=1` the output is int16 at the element index below, for
0-based `m,n` and `na = n/4`. The strides were measured across M in {4, 8, 16} and N in
{16, 32, 64}, and they scale with M, not N [HW sweep]:

```
slot(m,n) = 4·m  +  (na%4)  +  (na/4)·4M  +  (n%4)·16M
```

| component | stride |
|---|---|
| `m`          | 4    |
| `na%4`       | 1    |
| `na/4`       | 4·M  |
| `n%4` (lane) | 16·M |

The layout is bit-exact at N<=32 (`4×32×32`, `8×32×32`, `16×32×32`, `32×32×32`,
`12×64×16`). At
N>=64 the `n/16` term stops extrapolating linearly, and elements with n>=32 read wrong. So
this layout holds per task at N<=32. It is dense (M·N slots) exactly at N=32.

The layout came from a map probe, `ROCKET_INT16_PROBE=1` in `matmul_int16_rocket`. It feeds
inputs that make each `C[m,n] = m·N + (n+1)`, a unique and decodable signature. It then
reads the buffer as int8, int16 and int32 at once, which gives both the element size and
the slot-to-(m,n) map. **Trap:** the lane stride is `16·M`, and a probe at `N=4M` cannot
tell `16·M` from `N·4` because they coincide. Probe at least one shape with `N≠4M`.

## Full-precision int16

For an int64-exact result over full-range operands, decompose into int8.
`rocket_matmul_int16_exact` runs four int8 matmuls and recombines in int64, as
[tile-layouts.md](tile-layouts.md) describes. It costs four int8 matmuls where the int32
writer costs one int16 matmul, and it does not saturate. The two have not been timed
against each other.

## Unmeasured cases

For the int32 writer, none of these is measured:

- A K split across tasks
- A tiled or chained int16 job
- The multi-core path
- Speed against the byte decomposition
- The RK3576
