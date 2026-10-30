# CBUF operand reuse (CNA)

The CNA can skip the DRAM re-fetch of an operand tile that the previous task on the same
core left resident in the CBUF. Two bits in `CNA_CBUF_CON0` (0x1040), `WEIGHT_REUSE` and
`DATA_REUSE`, control it [source-confirmed: Mesa `rkt_regcmd.c`, HW sweep]. A reuse bit is
correct only while the whole run of tiles is one uninterrupted job on one core. The same
register contains `FC_DATA_BANK`, which the matmul datapath reads.

## Registers

| Offset | Field | Bits | Function |
|---|---|---|---|
| 0x1040 | `WEIGHT_REUSE` | 13 | Keeps the weight tile resident across tasks |
| 0x1040 | `DATA_REUSE` | 12 | Keeps the input-feature tile resident across tasks |
| 0x1040 | `FC_DATA_BANK` | 10:8 | Selects a CBUF data bank, and is live in conv mode (§"`FC_DATA_BANK[10:8]` on the matmul datapath") |

## Programming requirements

### Task order

The bits help only where consecutive *tasks batched into one job on one fd and one core*
share that operand. Order the tile loop to make the shared operand adjacent:

- **WEIGHT_REUSE**: iterate `(ni, ki, mi)`, so consecutive `mi` tasks reuse the same
  `(ni, ki)` weight. The reuse depth is `nMt`, the number of M-tiles.
- **DATA_REUSE**: iterate `(mi, ki, ni)`, so consecutive `ni` tasks reuse the same
  `(mi, ki)` input. The reuse depth is the per-worker `nNt`, the number of N-tiles for
  that worker.

Either loop order preserves the accumulation order, so the result is bit-identical to the
no-reuse path [HW sweep]. The reuse also works with the fp16 EW K-accumulation.

A 1-D task order makes only one operand "the same as the previous task". The weight tile and
the data tile cannot both be identical to the prior task while the third index still
advances. Pick the bit whose reuse depth is larger.

A task that streams its weight cube through the weight banks
([../matmul-as-conv.md](../matmul-as-conv.md) §"CBUF weight streaming") ends with only the
cube's last 32-kernel groups in the banks. A following task with `WEIGHT_REUSE` set would read
those groups as its whole weight tile [expected]. Software must clear `WEIGHT_REUSE` on a
streamed task.

### The one-job precondition

A reuse bit asserts that the operand the task wants is still in the CBUF. Anything else that
runs on that core between two tasks makes the assertion false, and a driver can schedule
another context's job between two submits. Software must set a reuse bit only within one
uninterrupted job.

Mainline `rocket` keeps a job's tasks together on the core, with no request from userspace.
`rocket_job_handle_irq()` programs the next task of the same job, and signals the done fence
only once `next_task_idx` reaches `task_count`. `core->in_flight_job` therefore holds the core
for the whole sequence [source-confirmed, v7.1].

The vendor BSP `rknpu` driver does not. Its job contains one program address:
`rknpu_job_commit()` programs `PC_DATA_ADDR` from `first_task->regcmd_addr` and never reads the
addresses of tasks 1..n-1. On `rknpu`, *n* unchained programs can only be *n* submits, hence
*n* jobs. `rknpu_job_next()` takes the next entry off `subcore_data->todo_list` the instant the
previous one retires.

> [!CAUTION]
> On the vendor `rknpu` driver, a reuse bit across unchained submits corrupts the output on
> roughly a quarter of runs [HW sweep, RK3588, `rknpu 0.9.8`]. Software must chain the tasks
> into one kick, or leave reuse off.

The corrupted output has the correct size, and its values change between runs. No error is
reported, and the job does not hang. The damage is one whole N-tile of one worker's column
slice (128 columns at `Nt=128`), filled with valid-looking numbers rather than zeros. The tile
did compute, against whatever operand the interleaving job left in the CBUF.

> [!NOTE]
> A test whose inputs are periodic does not detect the corruption. With a period-11 column
> pattern and 128-wide tiles, the wrong operand's bytes match the correct one's. The event
> occurs on roughly a quarter of runs, so a test needs tens of repeats, with the reuse-on and
> reuse-off configurations interleaved.

## `FC_DATA_BANK[10:8]` on the matmul datapath

`CNA_CBUF_CON0` (0x1040) also contains a 3-bit `FC_DATA_BANK` field at bits [10:8], above
`WEIGHT_BANK[7:4]` and `DATA_BANK[3:0]` [source-confirmed: Mesa `registers.xml`]. The field is
active in conv mode, which a matmul uses as a 1×1 conv [HW sweep, RK3588]. It is a
data-addressing field: a nonzero value points the feature read at the wrong CBUF bank. Software
must program `FC_DATA_BANK` to 0.

> [!CAUTION]
> Every `fc=1..7` corrupts the output of a matmul in conv mode [HW sweep, RK3588]. The result
> has ~64-75 % of its elements wrong. The error `|Δ|` is up to ~670 on values whose correct
> magnitude is a few thousand. The execution time does not change.

**Conditions:** RK3588, a 256×2048×1024 fp16 matmul. The execution time stays at ~30.2-30.8 ms
across all values, within noise.

> [!CAUTION]
> `fc=2..5` hang a task that streams its weight cube until the driver's watchdog retires it
> [HW sweep, RK3588, `rocket` 1.3.0, 2026-10-21].

The field likely selects a starting data bank that aliases once it runs past the populated
banks [hypothesis].

## Performance

**Conditions:** RK3588, Gemma prefill at pp2048, NPU clock ~816 MHz, fp16 EW K-accumulation on.

| Bit | Prefill gain |
|---|---|
| `WEIGHT_REUSE` | +1% (noise). The depth `nMt=2` is too small to matter at this ubatch |
| `DATA_REUSE` | +7% (13.5 -> ~14.5 t/s). The NPU `wait` bucket drops −21%, and every other bucket is flat |

The hardware honors `DATA_REUSE`, and the input-operand DMA it skips is not hidden behind MAC
compute. DATA_REUSE gains more because its reuse depth (per-worker `nNt`) is larger than
WEIGHT_REUSE's (`nMt`). Fewer, wider N workers deepen it further, which trades against
multicore N-splitting. DATA_REUSE removes a DMA, where most optimizations act on the
datatype-independent dispatch limit or on the clock
([../perf/not-mac-bound.md](../perf/not-mac-bound.md)).

## Software support

`librocketnpu` sets DATA_REUSE automatically with the fp16 K-accumulation, which is the
default operating mode. `ROCKET_REUSE` defaults to 2 whenever K-accumulation is on. Set
`ROCKET_KACC=0` to drop both.

`librocketnpu`'s matmul regcmd generator drives the CNA in conv mode (a matmul is a 1×1 conv)
and programs `FC_DATA_BANK` to 0.

RKNN's advertised `K<=10240` is an API-convenience window with no hidden single-pass trick.
`librocketnpu`'s tiler computes a `K=10240` matmul correctly, and its K-tiling already exceeds
the window. See [matmul-as-conv.md](../matmul-as-conv.md) §Tiling.

## Evidence

### Task order

The standalone test measures `max_abs=0.000` in both modes.

### The one-job precondition

Measured on one 128×1024×1024 fp16 matmul at 3-way fan-out, RK3588, `rknpu 0.9.8` [HW sweep]:

| Configuration | Corrupt runs |
|---|---|
| unchained submit, reuse on | 29 of 120 |
| unchained submit, reuse off | 0 of 120 |
| chained submit (one kick), reuse on | 0 of 80 |

Three independent single-threaded processes corrupt each other, which places the fault outside
any one process. A multicore test with a period-11 column pattern ran for months and reported
`max_abs=0.000` on every run.

### `FC_DATA_BANK[10:8]` on the matmul datapath

The test is `tests/fc_data_bank_sweep_rocket.c`. It forces the field to 0..7
(`ROCKET_FC_DATA_BANK` sentinel in `gen_matmul_task`). `fc=0` is byte-identical to the emitted
(unset) program. The test asserts that `fc=0` equals the unset program and that the default is
correct. It records `fc=1..7` as the known-live result, not a failure.

Flat time with corrupted data identifies a data-addressing, bank-selection field rather than a
performance setting. `fc>=2` saturate at the same `|Δ|`~667, and `fc=6` and `fc=7` give
identical diffs.

### Performance

A bit with no effect cannot reduce `wait`, so the −21% drop shows that the hardware honors
DATA_REUSE. The drop also shows that the input-operand DMA was not already hidden behind MAC
compute, because skipping a hidden DMA would change nothing.

### Software support

Part 2 of `tests/fc_data_bank_sweep_rocket.c` runs a single `K=10240` matmul (64×10240×256)
through `librocketnpu`'s tiler. The result is correct (`Kt=576`, `nKt=18`, `nbad=0` against
the fp32 CPU reference).
