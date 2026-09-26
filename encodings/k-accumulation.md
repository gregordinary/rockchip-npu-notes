# K-accumulation via the DPU eltwise unit (DPU-EW / DPU-RDMA)

When K is too large to contract in one CBUF-resident tile, the matmul is split into
`nKt` K-tiles whose partial products must be summed. One option is to read each
partial back and sum on the **host** in fp32, but that means reading every output
tile `nKt` times (`read ∝ M·N·nKt`). The other is to accumulate the K-partials
**on the NPU** using the DPU **eltwise (EW) add** path, reading each output tile once
(`read ∝ M·N`). For fp16 the on-NPU EW path is the **shipping default** (+19%, the
operating mode). The host fp32 sum is the byte-exact fallback/oracle (`ROCKET_KACC=0`).

**There is no on-chip third option: the conv accumulator cannot span tiles.** The conv's CACC
reduces K only *within one CBUF-resident tile*. The CORE register block has **no
accumulate-vs-reset control**, so nothing makes a later CBUF pass add into the prior CACC
contents. The conv task splitter splits spatial *height*, never the channel (K) axis
[source-confirmed: Mesa `rkt_task.c`].

So a K larger than one CBUF tile is always `nKt` separate jobs whose partials leave the chip.
The only choice is *where they are summed*: on the host (fp32 for fp16, int64 for the integer
types) or by the DPU-EW add below.

The EW add works for **fp16** and ships. No integer path ships. The recorded integer probes
failed, but they set only part of the precision bundle, so whether the EW adds int32 on this
silicon is unestablished. The integer lever would not move a prefill anyway, because the wall is
not readback.

## fp16 EW K-accumulation [HW sweep, source-confirmed]

The mechanism mirrors Mesa's working `add_tensor` residual-add geometry exactly. The
conv result is the DPU main input (MRDMA fed), and the ERDMA reads the running partial
from DRAM. The EW ALU adds them, and the WDMA writes back. Done ki-outer with a ping-pong
between two output BOs (in-place corrupts, because ERDMA would read the buffer WDMA is
writing).

The configuration that works (all fields are reg bits [4:31], i.e. the geometric
value `<< 4`):

- `DPU_EW_CFG = 0x108202C0`: **per-pixel** EW mode (bit28), `EDATA_SIZE=2` (16-bit),
  `EW_ALU_ALGO=2` (add), RELU/LUT bypass, `EW_OP_SRC=1`.
- `DPU_RDMA_ERDMA_CFG = 0x40000008`: ERDMA per-pixel mode (bit30), `DATA_SIZE=2`
  (16-bit fp16).
- `DPU_RDMA_FEATURE_MODE_CFG = 0x17D40`: `COMB_USE(5)` combines the conv main-data
  with the ERDMA operand (MRDMA *enabled* here, unlike the plain path).
- **`SURF_NOTCH = EW_SURF_STRIDE = MAX(out_w·out_h, 12) << 4`**: the planar
  pointer-advance. Leaving it 0 is the load-bearing trap: the ERDMA never advances,
  reads the offset-0 atom, and broadcasts it to every position/surface. (The
  `MAX(.,12)` floor over-states the stride for M<12, so **test with M>=12**.)
- ERDMA `EW_BASE = add_dma + out_w·out_h·16` (one surface offset, 16 B/position =
  8 fp16 channels = the atomic K block), while MRDMA `SRC_BASE = add_dma` (no offset).

The fp16 EW-add geometry above is HW-verified on this datapath. (Mesa's `add_tensor`
residual-add is int8/`EDATA_SIZE=1`, so its exact values don't transfer to the 16-bit fp16
path. See also the allbilly EW encodings in [SOURCES.md](../SOURCES.md).)

**Why single-knob sweeps mislead here.** Three registers must be right at once:
per-pixel (not per-channel) mode, a nonzero `SURF_NOTCH`/`EW_SURF_STRIDE`, and the right
ERDMA base. If any one is wrong, `EW_SURF_STRIDE` looks inert and the path looks
impossible, so no single-knob sweep can converge. Copy a known-good geometry (Mesa's
`add_tensor`) wholesale rather than sweeping one field at a time. With all three matching,
`N=16…384` (2->48 surfaces) and `M` up to 512 all pass.

**The precision cost.** The EW running sum accumulates *in fp16* (each add rounds),
whereas the host path sums fp16 partials in fp32. At real activation magnitudes this
is ~0.2-0.4% per matmul (max_abs ~8 on a 3840×4096, worst ~56 on the deepest FFN).

In practice it does **not** flip greedy LLM tokens. Gemma-4-12B output stays coherent,
and the per-op verify is `nonfinite=0`, worst `max_abs~0.38`. So fp16 EW K-accum is
"good enough," not bit-exact. **Measured: +19% Gemma prefill** (pp2048, 600 MHz).

## Integer EW K-accumulation

Per op, int8 without K-accum reads its int32 partials back at 2x fp16's bytes, so it is
strictly worse per op than fp16 with K-accum. EW is the only per-element SDP stage (BS/BN
broadcast a per-channel `[C]` vector, useless for per-pixel K-partials, see
[sdp-stage-precision.md](sdp-stage-precision.md)). Three approaches were probed through it, and
each failed:

1. **int32 integer-add via `EW_ALU_ALGO`** (`0x10C202C0`) -> garbage: the EW adds the
   int32 bit patterns *as float*.
2. **int32 via the `EW_OP_TYPE=1` integer path** (`0x10C203C4`) -> garbage: a true add of
   conv(226)+op(1000000) gives 1000226, and the EW returns `0x3A7C80`, an fp16 inf/NaN
   pattern in the low 16 bits. Tested with a **constant** operand, so it is the ALU, not
   the addressing.
3. **fp32 EW-add** (cast int8-conv's int32 -> fp32, then add via the float path). The pieces
   exist. The int8-conv -> fp32 output cast is bit-exact (still needs `size_e=7`). A 32-bit
   fp32 EW operand read works too (`EDATA_SIZE(3)`=32-bit, the same read the fp32-output matmul
   uses).

   But fp32 cannot hold the sum. A Kt=768 tile reaches ~12M and the full K-sum ~248M,
   past fp32's exact-integer range (2²⁴ ~16.7M). So accumulating the int32 partials as fp32
   drops the low bits. int16 EW-add cannot hold them either.

**Probes 1 and 2 do not establish a float-only EW ALU, and both are candidates for
partial-register-bundle negatives.** The TRM and an external silicon result both point the other
way. DPU `0x4010` (`RKNN_dpu_data_format`) enumerates `out_precision`, `in_precision` and
`proc_precision` identically and explicitly [TRM, `Rockchip RK3588 TRM V1.0-Part1`]:

| code | precision |
|---|---|
| `3'd0` | Integer 8bit |
| `3'd1` | Integer 16bit |
| `3'd2` | Float point 16bit |
| `3'd3` | Bfloat 16bit |
| **`3'd4`** | **Integer 32bit** |
| `3'd5` | Float point 32bit |
| `3'd6` | Integer 4bit |

There is a documented integer-32 processing precision. Probes 1 and 2 varied `DPU_EW_CFG`:
the ALU algorithm and the operand element size. They report that "the EW adds the int32 bit
patterns *as float*". That is exactly what a correct integer operand would produce in a stage
still carrying `proc_precision = 2` (fp16). Neither probe records setting the precision triple.

`allbilly/rk3588` reaches the opposite result on the same silicon [their HW, Orange Pi
RK3588, `examples/expt/elementwise_int.py`, not reproduced here, see SOURCES.md]. It measures
same-format integer **MAX, MIN, ADD, MINUS, ABS and NEG for INT8, INT16 and INT32**, across
widths 1/2/3 and partial and complete C1WC2 surfaces. INT8/INT16 saturate, INT32 ADD
saturates, and ABS/NEG preserve `INT32_MIN`. MUL passes for INT8/INT16, while its INT32 second
operand is signed 16-bit.

Their configuration writes the precision as a matched set rather than one field:

- `proc_precision`, `in_precision` and `out_precision` all `4` in DPU `0x4010`
- The same value again in `RDMA_FEATURE_MODE_CFG`'s input and proc fields
- `EDATA_SIZE` and ERDMA `DATA_SIZE` at the 32-bit code
- The EW converter bypassed with scale 1
- BRDMA and NRDMA explicitly disabled, so neither fetches through a stale address

**So "no on-device integer EW add" is not established, and it must not be cited as a hardware
ceiling.** Two things remain true independent of it. fp32 cannot represent an int32 K-sum
exactly (a Kt=768 tile reaches ~12M and the full K-sum ~248M, past 2^24), so the float route
was never the answer. And the int8 readback lever this was blocking was separately measured
not to move the wall, because the wall is not readback
([../perf/not-mac-bound.md](../perf/not-mac-bound.md)). The perf argument stands on its own.
The capability claim does not.

**Before reattempting, set the whole precision bundle, not the ALU algorithm.** A negative
established by moving part of a register bundle is not established. This IP has already forced
that rule on us for float mode, which is three registers that must move together.

## int4's int16 output: feasible in principle, but moot

int4's int16 output has K-sums that stay within fp32's exact-integer range, so int4 EW
K-accum via the float path *could* be bit-exact. But it doesn't matter: int4 can
reach single-pass K (`nKt=1`, zero K-accum) by shrinking the output tile, **and** the
matmul is not readback-bound anyway (single-pass int4 is no faster, see
[../perf/not-mac-bound.md](../perf/not-mac-bound.md)). So int4 EW K-accum is not worth
building: eliminating readback does not raise throughput here.

## Practical takeaway

- **fp16:** the EW K-accum (the `ROCKET_KACC` path) is the **default-on** operating
  mode. It gains +5-20% over the host-sum fallback across pp512-2048 on 0.8B/9B F16 (peak
  +20% at 9B pp512) [HW sweep 2026-06-28], with coherent greedy output. CBUF DATA_REUSE rides along
  automatically (~+7% more). Opt out with `ROCKET_KACC=0` for the byte-exact host sum.
- **int8/int16/int32:** no on-NPU K-accum ships, and whether the EW can do one is
  unestablished. Sum partials on the host (int64, bit-exact). To cut readback, grow Kt via the conv's native K-reduction (shrink
  Mt/Nt), but know it will not move the wall, because the wall is not readback.

### ki-fence chaining (`ROCKET_KACC_CHAIN`)

The KACC path fences once per ki-step (each ki>0 reads the prior partial). Chaining the
whole `[ki][tile]` sequence into one self-chained kick collapses nKt fences to one. The
HW honors the in-kick read-after-write, so it is **byte-exact** to the per-ki path (this is
how the in-kick-dependency property was proven, see
[regcmd-task-model.md](regcmd-task-model.md) §"in-kick data dependency"). But the ki-steps
are **serially dependent**, so a chained kick only pipelines the *independent* tiles within
each ki-block. Net effect tracks `gcap = BATCH/nKt` (BATCH=64) [HW sweep 2026-06-30, 600 MHz]:

| gcap = BATCH/nKt | example | vs per-ki |
|---|---|---|
| >=3 (nKt <= 21) | nKt=20 | ~0.95 (fence savings win) |
| 2 (nKt 22-32) | nKt=24,32 | ~1.01 (slight loss) |
| 1 (nKt >= 33) | nKt=40 | ~1.08 (`wait` +18%, serial stalls) |

So it is shipped **adaptive opt-in, default-off**: `=1` engages only at gcap>=3, else falls
back (never regresses). `=2` forces any fitting nKt (the byte-exact gate's strict gcap=1
test). Gemma FFN-down (nKt=40) falls back, so the end-to-end LLM gain is ~0. The value is
the in-kick-dependency finding + a regression gate, not a speed lever.
`tests/matmul_kacc_chain_rocket.c` (gate), `matmul_kacc_chain_bench.c` (A/B).

The diagnostic that cracked all of this is a standalone classifier
(`matmul_accum_rocket.c` / `matmul_accum_int8_rocket.c`) with a **constant** operand
and a sentinel output. It separates "can the EW add this type at all" from "is the
addressing right" in a single run. Keep it: it is the gate for any future EW attempt.
