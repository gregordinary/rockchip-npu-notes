<!-- The FA_THREADS correctness gate, ministral3-3b F16, RK3588 600 MHz, governor ondemand,
     taskset 0xf0 -t 4, llama-bench -p 2048 -n 0 -r 1 -v, one process per arm, cold page cache
     not reset between arms. Measured 2026-09-07 on the `.so` built from ggml-rocket HEAD plus
     the ROCKET_FA_CHECKSUM knob.
     THIS IS THE CORRECTNESS GATE, NOT A TIMING RESULT: one process per arm settles no sign on
     this board, and the governor is not pinned. The interval column is recorded because it was
     free, and it is read for its SHAPE only. -->

# ROCKET_FA_THREADS is bit-identical at every chunk count, measured

`ROCKET_FA_CHECKSUM=1` hashes every byte the `FLASH_ATTN_EXT` handler writes to its output,
FNV-1a with the standard 64-bit basis and prime, accumulated over every offloaded op, so the
value is reproducible outside this tree. Five chunk counts, one hash.

| `ROCKET_FA_THREADS` | checksum | ops | bytes | gather ms | scatter ms | G = g+s |
|---|---|---:|---:|---:|---:|---:|
| 1 | `7bbb7e8af60dd555` | 156 | 1308622848 | 3888 | 2781 | 6669 |
| 2 | `7bbb7e8af60dd555` | 156 | 1308622848 | 2324 | 1382 | 3706 |
| 3 | `7bbb7e8af60dd555` | 156 | 1308622848 | 1838 | 1246 | 3084 |
| 4 | `7bbb7e8af60dd555` | 156 | 1308622848 | 1511 | 967 | 2478 |
| 5 | `7bbb7e8af60dd555` | 156 | 1308622848 | 1561 | 963 | 2524 |

**`k`=3 is the one that matters.** It divides none of the walk lengths evenly, so it is the arm
that exercises the truncated last chunk and the early return for a pool worker above the chunk
count. `k`=4 divides evenly and would not have.

**The 12B unit agrees.** `k`=1 and `k`=4 on `gemma4-12b` F16 at pp2048 both read
`8852206bf3ccda62` over 288 ops and 2818572288 bytes, which is the gate `chain6` required before
it would spend an hour on the timing A/B. The equality also held on the earlier build, at its own
seed, so the split is checked on both binaries this session produced.

**`k`=5 is the clamp.** The pool has four workers on this part, so a request for five is clamped
to four, and the arm reads 1561 / 963 against `k`=4's 1511 / 967 -- the same interval within the
spread of a single process. A request above the pool's size buys nothing and costs nothing, which
is what the clamp is for.

**What the gate could not have shown.** It compares the handler's output to itself at another
chunk count, so it certifies the split and says nothing about whether the serial walk was right
to begin with. And it is one model and one shape: a walk whose outer index is degenerate at some
other geometry is outside what these five arms varied.

## The gate compares one run to one run, and that is not enough here

**Read [trackd37-fa-checksum-nondeterminism.md](trackd37-fa-checksum-nondeterminism.md) before
using any equality above as evidence.** Two runs of the 12B unit at ONE chunk count later
disagreed -- twice in seven runs on one build, same graph, same 288 ops and 2818572288 bytes --
so the handler is not reproducible run to run on this unit, and an equality across `k` is not
separable from whatever it does on its own. The runs recorded here happened; what they support
does not include the split. Re-establishing it needs the same-`k` disagreement rate bounded
first, then more repeats across `k` than that rate.

## Why the gate runs under llama-bench and not llama-perplexity

Perplexity is the more sensitive instrument and **it cannot be shown to reach this handler**. On
this model at `-c 2048`, `llama-perplexity` emits no `ROCKET FA total` and no `ROCKET FA checksum`
line at all, under `-fa auto` and `-fa on` alike, while `llama-bench` on the same model, `.so` and
prompt length emits both. It does reach the NPU: it holds `/dev/accel/accel0` for the whole run,
which the matmul offload alone accounts for.

**The cause is not established, and two candidates remain.** Either the scheduler never assigns
the `FLASH_ATTN_EXT` node to this backend under that tool's graph -- placement propagates to an FA
node from an adjacent assigned op, so it is a property of the whole graph and not of the op -- or
the node runs and its summary is lost, because both summaries are `GGML_LOG_INFO` lines emitted
from an `atexit` handler and `common_log`, which `common_init` installs and `llama-bench` does not,
is asynchronous and **discards messages once its worker is paused** (`common/log.cpp`, `add()`).
The discriminator is a perplexity taken with `ROCKET_FLASH_ATTN=0` against the default: the
handler's fp16 arithmetic is not the CPU tiled kernel's, so a value that moves says the handler
ran. **That has not been run.**

Either way the gate does not belong there: an instrument that cannot be shown to reach the code
under test certifies nothing when its arms agree. `llama-bench` does reach the handler, and the
checksum is the observable it otherwise lacks.

**Read the op count before the hash.** A run that reached no `FLASH_ATTN_EXT` op prints no line,
and two runs that print nothing agree about nothing.

## The interval's shape, read as a shape

Not a timing result: one process per arm, ondemand governor. What it does show is that the walks
respond to the chunk count and respond sublinearly -- `G_2`/`G_1` = 0.556 against an ideal 0.500,
`G_4`/`G_1` = 0.372 against an ideal 0.250 -- which is the direction the memory-traffic argument
predicts, since a strided copy's traffic does not scale with cores. The pinned 12B campaign is what prices it, and it reads 0.547 and 0.323
on the same axis [`trackd33-fathreads-12b.md`] -- close enough that the shape transfers between
two models and far enough that the number does not.
