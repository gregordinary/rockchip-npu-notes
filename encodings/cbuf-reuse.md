# CBUF operand reuse (CNA)

The CNA can **skip the DRAM re-fetch of an operand tile that is already resident in
the CBUF from the previous task on the same core.** Two bits in `CNA_CBUF_CON0`
control it [source: Mesa `rkt_regcmd.c`; HW-confirmed this project]:

- **bit 13, `WEIGHT_REUSE`**: keep the weight tile resident across tasks.
- **bit 12, `DATA_REUSE`**: keep the input-feature tile resident across tasks.

These only help if consecutive *tasks batched into one job on one fd/core* actually
share that operand. So you reorder the tile loop to make the shared operand adjacent:

- For **WEIGHT_REUSE**: iterate `(ni, ki, mi)`, so consecutive `mi` tasks reuse the same
  `(ni, ki)` weight. Reuse depth = `nMt` (number of M-tiles).
- For **DATA_REUSE**: iterate `(mi, ki, ni)`, so consecutive `ni` tasks reuse the same
  `(mi, ki)` input. Reuse depth = per-worker `nNt` (number of N-tiles for that worker).

The accumulation order is preserved either way, so the result is **bit-identical** to
the no-reuse path (standalone gate: both modes `max_abs=0.000`; composes cleanly with
the fp16 EW K-accum).

## "One job" is a precondition, not a description

*Tasks batched into one job* above is load-bearing, and it is a claim about the
**driver**, not about the tile order. The bit says "the operand you want is still in
CBUF"; anything else that runs on that core in between makes that false. A driver is
free to schedule another context's job between two submits, so the reuse bit is only
sound while the whole run of tiles is one uninterrupted job.

Mainline `rocket` gives that for nothing: `rocket_job_handle_irq()` programs the next
task of the **same** job and only signals the done fence once `next_task_idx` reaches
`task_count`, so `core->in_flight_job` holds the core for the whole sequence
[source-confirmed, v7.1]. Nothing in userspace has to ask.

The vendor BSP `rknpu` driver does not, and that is where this was measured. Its job
carries **one** program address (`rknpu_job_commit()` programs `PC_DATA_ADDR` from
`first_task->regcmd_addr` and never reads tasks 1..n-1's), so *n* unchained programs can
only be *n* submits, hence *n* jobs, and `rknpu_job_next()` takes the next entry off
`subcore_data->todo_list` the instant the previous one retires. Measured on one
128×1024×1024 fp16 matmul at 3-way fan-out, RK3588, `rknpu 0.9.8` [HW sweep]:

| arm | corrupt runs |
|---|---|
| unchained submit, reuse on | 29 of 120 |
| unchained submit, reuse off | 0 of 120 |
| chained submit (one kick), reuse on | 0 of 80 |

**What it looks like when it goes wrong.** A full, correctly sized, entirely plausible
output surface whose value moves between runs, never an error and never a hang. The damage
is one whole N-tile of one worker's column slice (128 columns at `Nt=128`), filled with
valid-looking numbers rather than zeros, because the tile really did compute, against
whatever operand the interleaving job left in CBUF. Three independent **single-threaded
processes** corrupt each other, which is what places the fault below any one process.

**Two traps around measuring it.** A gate whose inputs are *periodic* cannot see it: with
a period-11 column pattern and 128-wide tiles the wrong operand's bytes match the right
one's, and the multicore gate that had run for months reported `max_abs=0.000` on every
run. And a single run is not a measurement: the event fired on roughly a quarter of
runs, so an arm needs tens of repeats and the two arms need to be interleaved.

## You can only use one bit at a time

A 1-D task order makes only **one** operand "the same as the previous task." You
cannot have both the weight and the data tile be identical to the prior task while
still advancing the third index. So pick the bit whose reuse depth is larger.

## Measured: DATA_REUSE wins, +7% in-model

On Gemma prefill (pp2048, 600 MHz, on top of fp16 EW K-accum):

- **WEIGHT_REUSE -> +1% (noise).** The depth `nMt=2` is too small to matter at this
  ubatch.
- **DATA_REUSE -> +7%** (13.5 -> ~14.5 t/s), with the NPU `wait` bucket dropping **−21%**
  and every other bucket flat.

That **−21% wait drop is the proof the bit is hardware-honored**: a no-op bit cannot
reduce wait, and it also proves the input-operand DMA was *not* already hidden behind
MAC compute (otherwise skipping it would change nothing). DATA_REUSE wins because its
depth (per-worker `nNt`) is larger than WEIGHT_REUSE's (`nMt`); fewer-wider-N workers
would deepen it further (a tradeoff against multicore N-splitting).

This is one of the few levers that is **not** dtype-independent dispatch floor or
clock; it genuinely cuts a DMA. It rides along automatically with fp16 K-accum, which
is the **default-on** operating mode (`ROCKET_REUSE` defaults to 2 whenever K-accum is on;
set `ROCKET_KACC=0` to drop both).

## `FC_DATA_BANK[10:8]` is live on the matmul datapath

`CNA_CBUF_CON0` (0x1040) also carries a 3-bit **`FC_DATA_BANK`** field at bits [10:8],
above `WEIGHT_BANK[7:4]` and `DATA_BANK[3:0]` [source-confirmed: Mesa `registers.xml`].
Our matmul regcmd generator drives the CNA in **conv mode** (a matmul is a 1×1 conv) and
leaves `FC_DATA_BANK` = 0; the plausible assumption is that it is a *fully-connected-mode*
field and a don't-care on the conv path. **It is not** [HW sweep, RK3588].

Forcing it 0..7 (`ROCKET_FC_DATA_BANK` sentinel in `gen_matmul_task`) on a 256×2048×1024
fp16 matmul: `fc=0` is byte-identical to the emitted (unset) program, but **every
`fc=1..7` corrupts the output**: ~64-75 % of the result elements wrong, `|Δ|` up to ~670
on values whose correct magnitude is a few thousand, while the **wall time stays flat**
(~30.2-30.8 ms across all values, within noise). Flat time + corrupted data means the
field is a **data-addressing / bank-selection field that the conv (matmul) datapath
honors**, not a performance knob: a non-zero value points the feature read at the wrong
CBUF bank. (`fc>=2` saturate at the same `|Δ|`~667 and `fc=6`/`fc=7` give identical diffs,
consistent with the field selecting a starting data bank that aliases once it runs past
the populated banks.)

⇒ The generator **must keep `FC_DATA_BANK` = 0** (it does). This is recorded so the field
is not set speculatively as "the FC bank". On RK3588 in conv mode it is live and breaks
the matmul. Gate: `tests/fc_data_bank_sweep_rocket.c` (asserts `fc=0`==unset + default
correctness; characterizes `fc=1..7` as the known-live finding, not a failure).

**Large-K confirmation (same gate, part 2).** A single `K=10240` matmul (64×10240×256)
runs **correct** through our tiler (`Kt=576`, `nKt=18`, `nbad=0` vs the fp32 CPU
reference), confirming RKNN's advertised `K<=10240` is an API-convenience window with no
hidden single-pass trick; our K-tiling already exceeds it. See
[matmul-as-conv.md](../matmul-as-conv.md) §Tiling.
