<!-- The FA handler's two output states: what the disagreeing op read, how far its output moved,
     and a replay that reproduces it without the model. RK3588 RK1, rocket 1.3.0, 600 MHz,
     governor ondemand, power/control=auto, board otherwise idle. gemma4-12b F16, llama-bench
     -p 2048 -n 0 -r 1 -t 4 -v under taskset 0xf0, ROCKET_KACC=1, ROCKET_FA_THREADS=1,
     ROCKET_FA_CHECKSUM=3, ROCKET_FA_DUMP_OP=189, page cache dropped before every run, one
     process per run. libggml-rocket.so 15b391d2d447efbcb5c611b6c6f0b031 (the 2026-09-07 source
     of 138de09d plus the level-3 hashing and the dump, nothing else), librocketnpu.a
     1fd95700abdad3143afdd04c6458b5f1. Measured 2026-09-26 21:28-23:02 UTC.
     Raw: trackd40-fa-inputs.log. Harness: ../trackd40-fa-inputs.sh. Dump comparison:
     ../trackd40-fa-dump-diff.py. Replay: rocket-userspace tests/fa_replay_probe.c, built against
     the same librocketnpu.a. -->

# The FA handler's two states are one element one ulp apart, and the op's inputs are identical

## The rate

Twenty runs printed trackd39's two aggregate hashes and no third: `8852206bf3ccda62` fourteen
times and `345a87c827fa972b` six (runs 3, 5, 6, 12, 14, 15). Every run reports 288 ops. Every
odd run differs from run 1 in the same seven op lines trackd39 found, op 189 and its dataflow.

Over the three campaigns that is 13 of 53 runs.

## What op 189 read

At level 3 each op line carries the FNV-1a of the four dense fp16 tiles the driver is handed.
At op 189 the q, k, v and mask hashes are identical in all twenty runs, and only the output
hash differs. The raw dumps agree: q, k, v and mask are byte-identical across all twenty.

So the difference is made inside the handler or on the device, from identical inputs.

Op 45, the same layer and microbatch in the warm-up pass, reads different q, k and v. So it is
not a controlled twin of op 189, and op 189's being the only op that disagrees is a property of
its data rather than its position.

## How far the output moved

One element of 2,097,152: head 12, token 256, output channel 240. The fourteen modal runs hold
`0.80517578125` there and the six odd runs `0.8046875`, one fp16 ulp apart. Every other element
is byte-identical in all twenty runs. Token 256 is the first row of the second 256-row tile, and
channel 240 opens the last 16-channel group.

A float64 recompute from the dumped inputs, at the model's scale of 1.0 and no softcap, sits
about 7.5e-3 from both, the fp16 pipeline's own error. So it cannot say which state is right.

## The replay

`fa_replay_probe` feeds op 189's dumped tiles back through `rocket_flash_attn_fp16_ctx`, the
handler's own path, and compares every output with the two dumps. Every trial of every arm
below equals one of the two dumped outputs exactly, or, where a knob changes the arithmetic,
one of two outputs of its own.

| Arm | Workers | One context, 100 trials | Fresh contexts, 100 trials |
|---|---:|---|---|
| base, first process | 5 | 80 modal, 20 odd | 93 modal, 7 odd |
| base, second process | 5 | 83 modal, 17 odd | 13 modal, 87 odd |
| base | 1 | 100 modal | 100 modal |
| base | 2 | 100 modal | 9 modal, 91 odd |
| base | 3 | 63 modal, 37 odd | 100 modal |
| `ROCKET_KACC=0` | 5 | 93 modal, 7 odd | 16 modal, 84 odd |
| `ROCKET_REUSE=0` | 5 | 84 modal, 16 odd | 100 modal |
| `ROCKET_KACC_CHAIN=0` | 5 | 73 modal, 27 odd | 99 modal, 1 odd |
| `ROCKET_FA_CHAIN=0` | 5 | two outputs of its own, 74 and 26 | the same two, 77 and 23 |
| `power/control=on`, twice | 5 | 72 / 28, then 87 / 13 | 68 / 32, then 92 / 8 |

What this says:

- The flip needs concurrent workers. At one worker, 200 of 200 trials are modal.
- `ROCKET_KACC=0` gives the same two outputs as the base, so the AV contraction's
  K-accumulation is not in the path that differs. No knob tried removes the second state.
- The per-head path (`ROCKET_FA_CHAIN=0`) computes different values and still has two states.
- A context can hold one state for 100 trials (2 workers, one context) while fresh contexts
  mostly take the other. The workers are new threads on every call, so the state a context
  holds is in its fds or its buffers, not its threads.
- The rate is not a constant of the build: 7% to 91% between processes and arms.

## What this could not see

- Which NPU core ran which worker's jobs. The kernel spreads each fd's jobs over three cores
  and nothing here records where they landed. That the two states track core placement, or
  state a core keeps between jobs, is [hypothesis]; the `gpu_scheduler` trace events can read
  placement per job.
- Which of the two values is right.
- Whether other ops carry an element this close to a rounding tie that the runs have not
  drawn.
