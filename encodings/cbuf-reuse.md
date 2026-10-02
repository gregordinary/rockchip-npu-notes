# CBUF operand reuse (CNA)

The CNA can skip the DRAM re-fetch of an operand tile that the previous task on the same
core left resident in the CBUF. Two bits in `CNA_CBUF_CON0` control it
[source-confirmed: Mesa `rkt_regcmd.c`, HW sweep]:

- **Bit 13, `WEIGHT_REUSE`**: keeps the weight tile resident across tasks.
- **Bit 12, `DATA_REUSE`**: keeps the input-feature tile resident across tasks.

The bits help only where consecutive *tasks batched into one job on one fd and one core*
share that operand. So order the tile loop to make the shared operand adjacent:

- **WEIGHT_REUSE**: iterate `(ni, ki, mi)`, so consecutive `mi` tasks reuse the same
  `(ni, ki)` weight. The reuse depth is `nMt`, the number of M-tiles.
- **DATA_REUSE**: iterate `(mi, ki, ni)`, so consecutive `ni` tasks reuse the same
  `(mi, ki)` input. The reuse depth is the per-worker `nNt`, the number of N-tiles for
  that worker.

Either loop order preserves the accumulation order, so the result is bit-identical to the
no-reuse path. The standalone gate reads `max_abs=0.000` in both modes, and the reuse
composes with the fp16 EW K-accumulation.

## The one-job precondition

*Tasks batched into one job* above is a precondition, and it is a claim about the
driver, not about the tile order. The bit says "the operand you want is still in CBUF".
Anything else that runs on that core between two tasks makes that false. A driver is free
to schedule another context's job between two submits. So the reuse bit is sound only
while the whole run of tiles is one uninterrupted job.

Mainline `rocket` provides that unasked. `rocket_job_handle_irq()` programs the next task
of the same job, and signals the done fence only once `next_task_idx` reaches
`task_count`. So `core->in_flight_job` holds the core for the whole sequence
[source-confirmed, v7.1]. Userspace does not have to request it.

The vendor BSP `rknpu` driver does not, and that driver is where the failure was measured.
Its job carries one program address: `rknpu_job_commit()` programs `PC_DATA_ADDR` from
`first_task->regcmd_addr` and never reads the addresses of tasks 1..n-1. So *n* unchained
programs can only be *n* submits, hence *n* jobs. `rknpu_job_next()` takes the next entry
off `subcore_data->todo_list` the instant the previous one retires. Measured on one
128×1024×1024 fp16 matmul at 3-way fan-out, RK3588, `rknpu 0.9.8` [HW sweep]:

| arm | corrupt runs |
|---|---|
| unchained submit, reuse on | 29 of 120 |
| unchained submit, reuse off | 0 of 120 |
| chained submit (one kick), reuse on | 0 of 80 |

### Failure signature

The failure is a full, correctly sized, entirely plausible output surface whose value
moves between runs. It is never an error and never a hang. The damage is one whole N-tile
of one worker's column slice (128 columns at `Nt=128`), filled with valid-looking numbers
rather than zeros. The tile did compute, against whatever operand the interleaving job
left in CBUF. Three independent single-threaded processes corrupt each other, which places
the fault outside any one process.

### Measurement traps

A gate whose inputs are periodic cannot see the failure. With a period-11 column pattern
and 128-wide tiles, the wrong operand's bytes match the right one's. A multicore gate with
that pattern ran for months and reported `max_abs=0.000` on every run.

A single run is not a measurement. The event fired on roughly a quarter of runs, so an arm
needs tens of repeats, and the two arms must be interleaved.

## One reuse bit per task order

A 1-D task order makes only one operand "the same as the previous task". The weight tile
and the data tile cannot both be identical to the prior task while the third index still
advances. So pick the bit whose reuse depth is larger.

## In-model measurement

On Gemma prefill (pp2048, 600 MHz, with fp16 EW K-accumulation on):

- **WEIGHT_REUSE** gains +1% (noise). The depth `nMt=2` is too small to matter at this
  ubatch.
- **DATA_REUSE** gains +7% (13.5 -> ~14.5 t/s). The NPU `wait` bucket drops −21% and
  every other bucket is flat.

That −21% wait drop is the proof that the hardware honors the bit: a no-op bit cannot
reduce wait. It also proves the input-operand DMA was not already hidden behind MAC
compute (otherwise skipping it would change nothing). DATA_REUSE wins because its depth
(per-worker `nNt`) is larger than WEIGHT_REUSE's (`nMt`). Fewer, wider N workers would
deepen it further (a tradeoff against multicore N-splitting).

DATA_REUSE is one of the few levers that cuts a DMA. Most levers act on the
datatype-independent dispatch floor or on the clock. DATA_REUSE follows fp16
K-accumulation automatically, and fp16 K-accumulation is the default-on operating mode.
`ROCKET_REUSE` defaults to 2 whenever K-accumulation is on. Set `ROCKET_KACC=0` to drop
both.

## `FC_DATA_BANK[10:8]` on the matmul datapath

`CNA_CBUF_CON0` (0x1040) also carries a 3-bit `FC_DATA_BANK` field at bits [10:8],
above `WEIGHT_BANK[7:4]` and `DATA_BANK[3:0]` [source-confirmed: Mesa `registers.xml`].
`librocketnpu`'s matmul regcmd generator drives the CNA in conv mode (a matmul is a 1×1
conv) and leaves `FC_DATA_BANK` = 0. The plausible assumption is that the field belongs to
a fully-connected mode and that the conv path ignores it. **The conv path honors it**
[HW sweep, RK3588].

The probe forces the field to 0..7 (`ROCKET_FC_DATA_BANK` sentinel in `gen_matmul_task`)
on a 256×2048×1024 fp16 matmul. `fc=0` is byte-identical to the emitted (unset) program.
**Every `fc=1..7` corrupts the output.** The result has ~64-75 % of its elements wrong.
The error `|Δ|` is up to ~670 on values whose correct magnitude is a few thousand. The wall
time stays flat (~30.2-30.8 ms across all values, within noise).

Flat time with corrupted data means the field is a data-addressing, bank-selection field
that the conv (matmul) datapath honors, not a performance knob. A non-zero value points the
feature read at the wrong CBUF bank. `fc>=2` saturate at the same `|Δ|`~667, and `fc=6`
and `fc=7` give identical diffs. Both observations are consistent with the field selecting
a starting data bank that aliases once it runs past the populated banks.

**The generator must keep `FC_DATA_BANK` = 0**, and it does. Do not set the field
speculatively as "the FC bank". On the RK3588 in conv mode it is live and breaks the
matmul. The gate is `tests/fc_data_bank_sweep_rocket.c`. It asserts that `fc=0` equals the
unset program and that the default is correct. It characterizes `fc=1..7` as the
known-live finding, not a failure.

### Large-K run in the same gate

Part 2 of the same gate runs a single `K=10240` matmul (64×10240×256) through
`librocketnpu`'s tiler. The result is correct (`Kt=576`, `nKt=18`, `nbad=0` against the
fp32 CPU reference). That confirms RKNN's advertised `K<=10240` is an API-convenience
window with no hidden single-pass trick. `librocketnpu`'s K-tiling already exceeds it. See
[matmul-as-conv.md](../matmul-as-conv.md) §Tiling.
