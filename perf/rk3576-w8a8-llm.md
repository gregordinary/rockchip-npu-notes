# W8A8 LLM prefill on the RK3576

`ggml-rocket`'s RK3576 W8A8 path runs an LLM's projections on the RK3576's int8 matmul from stock
llama.cpp. The host quantizes each weight per output channel and each activation to int8, and
rotates both with a Hadamard transform along `K`. `rocket_matmul_int8_rk3576_perc()` then writes
int8 through a per-column requant. Its scale comes from a per-column magnitude estimate,
calibrated over the first forward passes. On Qwen2.5-1.5B its prefill is 1.02x to 1.31x the CPU's
Q8_0 kernel, and 2.01x to 2.37x the CPU on the same F16 GGUF. Its wikitext-2
perplexity is 1.011x the CPU's [HW sweep, H96 MAX M9].

**Conditions:** H96 MAX M9, `rocket` 1.6.0, Qwen2.5-1.5B F16 GGUF, `ROCKET_INT8=1`, four threads
pinned to the A72s, governor `performance`, one job in flight. `pp512` at ubatch 512 on llama.cpp
b10356, measured 2026-08-11, where every offloaded call is a calibration forward. The run
predates the calibration scan's row-major form (§"The calibration scan's loop form"), the
supplied weight sums and the resident weight cube (§"Supplied sums, the resident weight cube and
the BO pool").
`pp2048` at `-ub` 512 on llama.cpp b10558 with the `ggml-rocket` build of 2026-08-19, measured
2026-08-23. Perplexity over the first 8 chunks of wikitext-2 test at `n_ctx` 512, measured
2026-08-11. The NPU clock was not read during these runs, and every read on this board before
2026-09-30 is 786 MHz [live 2026-07-25 to 2026-09-25].

| Prompt | Against the CPU on Q8_0 | Against the CPU on F16 |
|---|---:|---:|
| `pp512` | 1.02x | 2.01x |
| `pp2048` | 1.31x | 2.37x |

The F16 ratio is the one a drop-in user sees, because the path consumes an F16 GGUF. In host
simulation, with one per-tensor output scale and no rotation, the same entry raises a model's
perplexity 1.11x to 2.26x [host-computed]. The int8 entry's datapath, its bounds and its
per-call cost are in [../chips/rk3576.md](../chips/rk3576.md).

## The W8A8 path against the CPU at `pp512`

Under stock llama.cpp on Qwen2.5-1.5B, the W8A8 path ties the strongest CPU configuration and
doubles the weakest. It is 1.02x the CPU on a Q8_0 GGUF and 2.01x the CPU on an F16 GGUF, at
`pp512`. [HW sweep, H96 MAX M9, 7.1.7 / `rocket` 1.6.0, llama.cpp b10356, measured 2026-08-11]

**Conditions:** H96 MAX M9, kernel 7.1.7, `rocket` 1.6.0, llama.cpp b10356, measured
2026-08-11. `pp512` at ubatch 512, four threads pinned to the A72s, governor `performance`, one
job in flight. The NPU configuration runs the `ggml-rocket` DL backend with `ROCKET_INT8=1`. Its
GGUF is an F16 conversion of the same HF snapshot the accuracy corpus is scored on. The CPU
configurations ran in the same boot, interleaved with it. The `ggml-rocket` build predates the
calibration scan's row-major form, the supplied weight sums and the resident weight cube.

| Configuration | GGUF | pp512 t/s (two processes) | NPU speedup over it |
|---|---|---:|---:|
| CPU only | F16 | 5.92, 5.77 | **2.01x** |
| CPU only | Q8_0 | 11.58, 11.51 | **1.02x** |
| NPU, W8A8 | F16 | 11.78 +- 1.80, 11.78 +- 1.84 | n/a |

The port consumes an F16 GGUF, and on the CPU that GGUF runs on ggml's F16 `MUL_MAT`. The 2.01x
is therefore the figure a drop-in user sees, and it includes the CPU's cost of that input format.
The 1.02x is against ggml's Q8_0 kernel, the baseline of the port's 2.37x model-weighted cap
([../chips/rk3576.md](../chips/rk3576.md#the-int8-gemm-against-a-real-cpu-gemm-at-llm-prefill-shapes)
§"The int8 GEMM against a real CPU GEMM, at LLM prefill shapes").

The 2.37x cap overstates the end-to-end port by 2.3x. It is a cap over projection MACs, on shapes
this model does not have. Three things separate it from the end-to-end result:

- Every offloaded call here is `K`=1536, and none of the six shapes behind the cap is.
- The down-projection's 29.4% of projection MACs runs on the CPU, by the library's own
  `K >= 6176` refusal. The cap does charge that part.
- The wall-clock time also includes attention, the norms, rope, softmax and the scheduler's
  copies.

Six of the seven projections per layer are offloadable, 168 over 28 layers, and the frontend
offloads 166 of them per micro-batch [HW sweep 2026-08-11]. Under `llama-bench` the final block's
gate and up run at `M`=1, on the one output row the bench requests, below the frontend's minimum
`M` [host-computed]. The down-projection and the `lm_head` (computed at `M`=1 for the last token
only) stay on the CPU.

The `performance` governor is worth 16% to the NPU configuration, which measures 9.87 and
9.91 t/s on `ondemand` against 11.78 pinned.

> [!NOTE]
> The NPU configuration's run-to-run spread is +-1.8 t/s (15%), against +-0.01 on both CPU
> configurations. A single process does not measure it.

`ROCKET_MM_PROFILE` splits this path's time with its own accumulator. The RK3588 accumulators
`g_prof` and `g_prof_i8` in `rocket_matmul.c` split an RK3588 job into
pack/gen/sync/submit/wait/read. Three of this path's terms have no bucket there: the O(K*N)
`sum_abs_w` pass, the per-tile per-column ramp plan, and the write guard's stamp-and-check.

The path's own buckets in `rocket_matmul_rk3576.c` are
`sumabs / packA / packB / coeff / gen / alloc / stamp / submit / wait / read`, over calls,
tiles and submits. On the same knob, `ggml-rocket.cpp` reports the terms outside the entry
(`valloc / quant+rot / calibrate / entry / dequant`). Both sum to the offloaded path, not to the
prefill wall-clock time. Attention, the norms, rope and the scheduler are in neither.

## Flash attention past a 512-token prompt

With flash attention on, the NPU configuration fails at prompts of 1024 and 2048 tokens: it
returns `failed to decode prompt batch, res = -3`, and llama.cpp stops. The refusing op is
`GGML_OP_FLASH_ATTN_EXT`, and the trigger is `n_kv >= 1024`. The micro-batch does not move it, so
the axis is the context, not the ubatch.

**Conditions:** H96 MAX M9, Qwen2.5-1.5B F16, llama.cpp b10356, measured 2026-08-11. One
process at a time, on an otherwise idle board.

**Not characterized:** a prompt between 512 and 1024 tokens, and the end-to-end effect of
`-ub 2048` on the RK3576.

| Configuration | `-b`/`-ub` | prompt | result |
|---|---|---|---|
| NPU, FA on | 512 / 512 | 512 | **rc=0**, 10.51 t/s |
| NPU, FA on | 512 / 512 | 2048 | `res = -3` after 79 s |
| NPU, FA on | 1024 / 1024 | 2048 | `res = -3` after 2 s |
| NPU, FA on | 2048 / 2048 | 2048 | `res = -3` after 4 s |
| NPU, FA on | 1024 / 1024 | 1024 | `res = -3` after 5 s |
| NPU, FA off | 512 / 512 | 2048 | **rc=0**, 12.12 t/s |
| NPU, FA off | 1024 / 1024 | 2048 | **rc=0**, 12.10 t/s |

[HW sweep, H96 MAX M9, Qwen2.5-1.5B F16, llama.cpp b10356, measured 2026-08-11]

A ubatch of 512 fails at a 2048-token prompt, and the passing control computes that micro-batch
shape 168 times per ubatch. The context is therefore what separates the passing row from the
failing ones. The elapsed times locate the failure inside the run. At ubatch 512 it comes after
79 s, in the second micro-batch, where the KV length first crosses 1024. When the first
micro-batch is already that long, it comes after 2-5 s.

The `ggml-rocket` source gives the cause. A `-3` from `llama_decode` is `GGML_STATUS_FAILED`. An
allocation failure is a distinct `-2` (`ALLOC_FAILED`), so the board's RAM is not the cause. In
`ggml-rocket`, `supports_op` claims `FLASH_ATTN_EXT` subject only to shape, type,
`n_tokens >= 16` and `n_kv >= rocket_flash_attn_min_kv()` (default 1024). Unlike its `MUL_MAT`
case, whose comment states this exact failure mode, that case has **no `rocket_rk3576_selected()`
branch**.

The handler's only compute call is `rocket_flash_attn_fp16*`, and every path in `rocket_attn.c`
goes through `rocket_matmul_fp16`. That function begins with `mm_wrong_encoding()`, and returns
`ROCKET_E_UNSUPPORTED` on any profile that is not `rk3588`. The flash-attention handler therefore
reaches the RK3576 unadapted. At a prompt of 512 the `n_kv` threshold kept it out of reach, which
is why the whole corpus ran at that length.

The backend claims the op and then declines it, and the scheduler has nowhere to put it. It is a
coverage gap in the backend, not a hardware bound. Declined in `supports_op` on the RK3576,
`FLASH_ATTN_EXT` goes to the CPU. `ROCKET_FLASH_ATTN=0` already does that, and makes the port
usable past a 512-token prompt.

> [!NOTE]
> Under a bench harness the refusal's message is not printed. Both the `ROCKET_LOGE` in
> `mm_wrong_encoding()` and the `GGML_LOG_ERROR` in `graph_compute` fire. The `ggml-rocket`
> backend forwards the rocket channel into ggml's, and `llama-bench` installs a no-op ggml log
> callback that discards it. `ROCKET_DEBUG=1` raises the level and does not change the sink. Set
> `ROCKET_LOG_STDERR=1`, which the convert, int8 and MoE profiler dumps use for the same reason.
> Under a bench harness, read `res = -3` as a message that is not being shown.

With flash attention declined, the port measures 12.12 t/s at ubatch 512 and 12.10 at
ubatch 1024. Doubling `M` per offloaded call moved the wall-clock time 0.2%, with one process per
test point against the configuration's +-15% run-to-run spread. Runs that vary only the ubatch
also move the context with it. The profile buckets below measure the effect, and two wall-clock
points cannot.

## Time and wall-clock share of the offloaded W8A8 path

The offloaded W8A8 path is host-bound. The `wait` bucket, the share that runs programs, is 13.7%
of the entry's accounted time at `M`=512, 19.7% at 1024 and 25.6% at 2048. The two largest
buckets are `alloc` (30.7% at `M`=512) and `packB` (21.7%). The host is the cost, and the NPU is a
quarter of it at best. [HW sweep, H96 MAX M9, rocket 1.6.0, measured 2026-08-11]

**Conditions:** H96 MAX M9, `rocket` 1.6.0, Qwen2.5-1.5B F16 on llama.cpp b10356, measured
2026-08-11. `ROCKET_MM_PROFILE=1 ROCKET_LOG_STDERR=1` and `ROCKET_FLASH_ATTN=0`, with `-ub` swept
over {512, 1024, 2048} at a fixed prompt of `-p 2048`, and the governor pinned to `performance`.
Every test point runs the same total tokens and the same total MACs, with the per-call count
quartered.

The model offloads 166 matmuls per micro-batch, all at `K`=1536. Across the whole axis
the entry's buckets are constant at 2.301 tiles per call. Submits per call double with `M`: 9.20
at `M`=512, 18.41 at 1024 and 36.82 at 2048. The `alloc` bucket is large because the weight,
coefficient and output BOs are allocated and freed per tile per call.

Per library call, in ms:

| bucket | M=512 | M=1024 | M=2048 | scaling |
|---|---:|---:|---:|---|
| `sumabs` | 3.62 | 3.61 | 3.61 | M-independent |
| `packB` | 7.51 | 7.58 | 7.60 | M-independent |
| `coeff` | 0.219 | 0.218 | 0.215 | M-independent |
| `gen` | 0.044 | 0.068 | 0.116 | ~M-independent |
| `packA` | 1.87 | 5.35 | 11.32 | M |
| `alloc` | 10.63 | 13.33 | 18.74 | mixed |
| `stamp` | 2.13 | 5.29 | 14.51 | M |
| `submit` | 0.148 | 0.246 | 0.449 | M |
| `wait` | 4.75 | 10.50 | 24.88 | M |
| `read` | 3.65 | 6.96 | 15.60 | M |
| **accounted** | **34.57** | **53.16** | **97.05** | |

The `M`=512 column reproduces on an unchanged binary between 2026-08-11 and 2026-08-13. Every
bucket sits inside 5.3%, and all but `packA` and `read` inside 1.5%.
[HW sweep, H96 MAX M9, 2026-08-13]

### The M-independent intercept

On the per-call path, where the entry recomputes per call, the M-independent intercept is
11.4 ms per call. The figure is measured directly, not fitted: `sumabs+packB+coeff+gen` measures
11.39, 11.48 and 11.54 ms across a 4x sweep of `M`, constant to 1.3%. It is three passes over
`K*N` every call, and 31.3% of the converged per-call cost at `M`=512. The default frontend path
does not run them. Its caller hands the entry the sums (`_sa`) and the weight resident in a
device cube (`_wbo`, below).

The intercept is therefore the cost of the calibration forwards, and of any caller still on the
per-call path. It is `sumabs` plus `packB` and essentially nothing else. In the 2026-08-13
measurement, the `coeff` bucket is 0.216 ms a call and `gen` is 0.045, together 0.8% of the
11.4. `sumabs` and `packB` together are 11.136 ms = 32.35% of the accounted total. The `alloc`
bucket (10.640 ms, 30.91%) is outside this intercept.

### Supplied sums, the resident weight cube and the BO pool

One weight BO serves every N tile and every `M`, because the whole-`N` int8 weight cube is the
per-tile cubes concatenated
([../chips/rk3576.md](../chips/rk3576.md#the-matmul-entrys-requant-and-per-column-ramp)).

The default frontend path supplies the per-column weight sums and keeps the weight resident in
a device cube. Two M-independent buckets, `sumabs` and `packB`, therefore do not run. With the
BO pool as well, the path's accounted cost a call falls 59%. Prefill throughput at `pp2048` rises 12.30 ->
13.41 t/s with the resident cube, and to 13.53 t/s (+10.0%) with the cube and the pool. Every
configuration is bit-identical to the per-call path [HW sweep, H96 MAX M9, 2026-08-19].

**Conditions:** H96 MAX M9, Qwen2.5-1.5B F16 on llama.cpp, `-ub 512`, one binary, measured
2026-08-19.

**Not characterized:** the IOMMU churn on kernels without the BO-lifetime fixes, behavior under
real memory pressure, and decode at M=1. At M=1 these changes are expected to remove that
call's 77% `packB`+`alloc` share [expected].

Per call, in ms:

| bucket | control | resident | resident+pool |
|---|---:|---:|---:|
| `packB` | 7.49 | 1.66 | 0.95 |
| `alloc` | 10.60 | 5.18 | 0.015 |
| `bofree` | 1.32 | 0.84 | 0.004 |
| `packA` | 1.84 | 1.59 | 0.98 |
| accounted total | 32.13 | 19.89 | **13.06 (-59%)** |

The `packB` residue is the 332 calibration forwards, which still take the host path. A fresh
`CREATE_BO` faults in demand-zero pages, and those faults spread across the first-touch
buckets, so the pool removes more than its own `alloc`. It also cuts `calibrate` 21.2 -> 11.4 s
a process, because the bootstrap's create faults fell in that bucket. With the host stalls
removed, the `wait` bucket rises 14.1 -> 15.9 s, and the NPU becomes the pacing element
[hypothesis: queuing]. [HW sweep, H96 MAX M9, 2026-08-19]

The per-column sum of |B| that caps the C ramp is a quantity `ggml-rocket` already computes once
per weight, as the source of `cbound`. The entry `rocket_matmul_int8_rk3576_perc_sa()` takes
that sum instead of recomputing it, so the `sumabs` bucket does not run.
`ROCKET_RK3576_WSA=0` restores the recomputing form, to measure its cost. The two forms are
bit-identical by construction, and on the device [HW sweep].

The function `rocket_rk3576_wbo_create()` packs a weight once into a device BO. The entry
`rocket_matmul_int8_rk3576_perc_wbo()` is `_sa` with that object in place of `B`. It requires
`sum_abs_w`, refuses a K/N mismatch, and refuses outright a `K` that has no plan. The frontend
creates the object at a weight's first converged call and drops its host `qB`. Device memory
therefore goes up by what host memory comes down.

`ROCKET_RK3576_WDEV` is on by default, and `=0` restores per-call packing. Calibration still
reads `qB` and still takes the per-call path.

`ROCKET_RK3576_BO_POOL=1` keeps the path's transient BOs on a per-fd free list capped by
`ROCKET_RK3576_BO_POOL_MB`. The variable is read per allocation. With the virtual layout pinned by `setarch -R`, the pool raises `pp2048` throughput by +1.86%, 13.9800
against 13.7250 t/s [HW sweep, H96 MAX M9, 2026-08-20]. The +1.86% is the effect at the modal
layout, not the expected value with ASLR on, and the pool stays off by default.

The pool's gain is DRAM traffic, not host CPU time. The pooled configuration's
`l2d_cache_refill` level is -1.50% against the unpooled one's (16166.3M vs 16412.6M). The
reduction is the same order as the throughput gain [HW sweep, H96 MAX M9, 2026-08-20]. It is
in the same direction as the L2 refill term behind the per-process variation below.

### Per-process variation in prefill throughput

Prefill throughput on this board varies between processes by a term fixed when each process
starts, and the virtual address layout sets it. Repetitions inside one process agree to
+-0.25%. Across 30 processes from seven measurement series, 26 sit within 0.30% of their group
median and 4 sit 2.8-3.6% out. An empty gap separates the two groups. One of the four is
fast, so the modal cluster is not the best case [HW sweep, H96 MAX M9].

**Conditions:** H96 MAX M9, Qwen2.5-1.5B F16 on llama.cpp, `pp2048` unless a paragraph names
`pp512`, four threads on the A72 cluster (`taskset -c 4-7`), governor `performance`.

The slow processes spend their excess between the offloaded calls, as DRAM traffic from L2
misses. In a slow process `l2d_cache_refill` is +23.7% (+4.0e9), with instructions +0.05% and
`l1d_cache_refill` +0.0%. Cycles rise +4.4%, tracking the wall-clock time
[HW sweep, H96 MAX M9, 2026-08-19].

Pinning the virtual layout with `setarch -R` removes the slow draws. Over 16 interleaved
`pp2048` test points it takes the `l2d_cache_refill` CV from 9.699% to 0.099%, and the
throughput spread from 4.966% to 0.292% (13.69-13.73 t/s). The pin selects the modal layout
rather than creating a new operating point: the two configurations' medians are 13.705 and
13.720 t/s. With `setarch -R` and every test point in one invocation, the process-to-process sd
of throughput is 0.087% [HW sweep, H96 MAX M9].

Physical frame identity does not set the throughput level. Anonymous memory gets a fresh set of frames in
every process [HW sweep, H96 MAX M9]. The GGUF's page cache keeps byte-identical frames across
every invocation. `setarch -R` does not bound a component between invocations. Under it,
one `pp512` level moved +2.508% between two invocations separated by other work, while the
interleaved ASLR-on configuration did not move. The candidate for that component is the
physical or system state the allocator hands out [hypothesis].

A second per-process draw, independent of the first, sets the speed of the weight-convert
loops. The `wt_rotate` loop takes 10753 ms or 11927 ms (1.109x) for byte-identical work, at a
~50% base rate. A scalar `wt_quant` loop takes the same mode in the same process. The draw
falls in weight conversion, outside the timed repetitions, so it does not move `pp2048`
throughput. `setarch -R` pins it at 11907-11923 ms, 0.15% slower than the ASLR-on modal cluster
and 10.8% slower than the low state [HW sweep, H96 MAX M9].

Backing the multi-MB anonymous buffers with 2 MB pages lowers prefill throughput 0.64% and does
not move the L2 refill count. The kernel has no THP (the sysfs node is absent), so the pages
come from hugetlbfs through glibc 2.42's `glibc.malloc.hugetlb=2`. Over five interleaved pairs
the hugetlb configuration measures 13.600 t/s against 13.688 [HW sweep, H96 MAX M9]. No process
in that series drew the slow state, so whether 2 MB pages blunt the term when it occurs is not
characterized.

> [!NOTE]
> On this board, a `pp2048` difference under ~5% between two processes is not a measurement,
> whatever their order. Compare configurations under `setarch -R` in one invocation, or
> interleave them and count the slow draws. A CV over 3-5 processes shows whether a slow draw
> fell in the sample, not the term's size. Read the per-process sd from each series' own
> controls. The per-call buckets hold to 2-3.5% across sessions, so compare a change at its
> bucket.

> [!NOTE]
> Match the `wt_rotate` and `wt_quant` modes before comparing those buckets across builds. The
> convert-loop mode does not predict a slow throughput draw.

### L1d set congruence across rebuilds

On the A72, a rebuild can change an untouched hot loop's speed 2.2x through L1d set
congruence [HW sweep, objdump, 2026-08-19]. A frontend change confined to the
weight-convert path moved the quantize loop `q8` from 9942 to 21870 ms over the same 2656
calls. The two binaries' q8 loops are instruction-identical: the same 11 instructions, the same
registers, and one `lrintf@plt` call per element.

**Conditions:** H96 MAX M9, A72 cluster, Qwen2.5-1.5B F16, ubatch 512, measured 2026-08-13 and
2026-08-19.

The whole-process A72 counter pair measures instructions -0.18%, `l1d_cache_refill` +8.42% and
`l2d_cache_refill` +0.11%. The extra L1d misses hit in L2 with no added instructions. The cost is
L1d conflict, not footprint [HW sweep, objdump, H96 MAX M9, 2026-08-19]. The
A72's L1d is 32 KiB 2-way. The loop interleaves a float read stream and a byte write stream
with a per-element libm stack frame. The binary's frame sizes and startup allocation sequence
set the three-way congruence [hypothesis].

The congruence is a property of the build, and any translation-unit or link change can change
it. The same frontend change rebuilt against a newer library measures `q8` at 9.9-10.0 s, the
fast level [HW sweep, objdump, H96 MAX M9, 2026-08-19].

> [!NOTE]
> On the A72, a rebuild can move a loop it did not touch by up to 2.2x. Compare a bucket across
> two builds only beside a same-binary control.

The structural fix for a loop in this position is a NEON `fcvtns` quantize (bit-exact
`lrintf`, ties to even), which removes the per-element call the congruence depends on. The
quantize loops are NEON lanes. Both branches of `rk76_quant_act` and `rocket_quant_wt_int8_had`
quantize by `fmul`, a float pre-clamp to +/-127, `fcvtns` and saturating narrows, with amax by
`vabs`/`vmax`. The `fcvtns` instruction rounds to nearest ties-to-even exactly as `lrintf` does
under the default rounding mode. The pre-clamp makes the narrows match the scalar's integer
clamp on every value, including the +/-127.5 ties. The lane form is therefore bit-identical by
construction [HW sweep, H96 MAX M9, 2026-08-19, `ggml-rocket` 1210ea8].

The `--chunks 4` equality test point prints PPL 8.7433 to every digit. In three w1p0-shape test
points, `quant+rot` goes from 30.9, 30.8 and 32.5 s a process to 19.56, 19.73 and 19.63 s. The
`wt_quant` bucket goes from 5.80, 5.80 and 6.56 s to 0.976 s in all three
[HW sweep, H96 MAX M9, 2026-08-19].

A loop with no per-element call gives the three-way congruence nothing to form on [expected
over future epochs, design intent, not yet a measurement]. The loops the change did not convert
keep the exposure. The `wt_rotate` loop moved +7.3% across the same rebuild with untouched
instructions [hypothesis: the same class's small end]. [HW sweep, H96 MAX M9, 2026-08-19]

### Share of prefill wall-clock time

At ubatch 512 the offloaded W8A8 path is 21.8% of the prefill wall-clock time. The timed pass
runs 664 converged calls, so the path is 37.1 s of a 169.7 s prefill [HW sweep, one test point,
2026-08-11]. The other 78% is outside the frontend's matmul handler, and the per-call buckets
cannot see it.

**Conditions:** H96 MAX M9, Qwen2.5-1.5B F16, `-p 2048`, `ROCKET_FLASH_ATTN=0`, read from the
timed pass at ubatch 512, one test point, measured 2026-08-11.

| term | per converged call, M=512 | per converged call, M=2048 |
|---|---:|---:|
| `valloc` | 0.41 | 7.47 |
| `quant+rot` | 16.99 | 67.07 |
| `entry` | 36.40 | 99.00 |
| `dequant` | 2.01 | 8.01 |
| **offloaded path** | **55.81** | **181.55** |

Measured standalone at `K=N=2048`, the M-independent intercept is 39% of the entry's own
wall-clock time. It is 31.3% of a converged call in the model at `M`=512.
Inside a model the entry is 14.2% of the prefill wall-clock time, 664 converged calls of 36.40 ms
in a 169.7 s prefill. The same term is therefore 4.5% of prefill [host-computed].

An inter-call gap accumulator, dumped as `gap=` on the frontend profile line, sums the
exit->entry gaps between handler calls. Its first test point measured gap 554.2 s beside
149.5 s of buckets over 2656 calls (78.7%), closing the span. The per-process variation above is
therefore directly observable per process [HW sweep, one test point, 2026-08-13].

Raising the micro-batch to `-ub 2048` is worth 4.1% of prefill [host-computed]. Per weight per
2048 tokens, the offloaded path costs 4 x 55.81 = 223.2 ms at ubatch 512 against 1 x 181.55 at
ubatch 2048. The cut is 18.7%, and over 166 weights it is 37.05 s against 30.14 s on a
169.7 s prefill. The entry-side cut alone is 32.0% (145.6 -> 99.0 ms). A two-point throughput
comparison cannot resolve 4% against this configuration's +-15% spread.

The `valloc` bucket, three value-initialized scratch vectors, is 0.59% of the offloaded path at
`M`=512 and 2.7% at `M`=2048. Its cost is superlinear in `M`, which is consistent with fresh
pages being faulted rather than with the memset itself.

### Prefill time by function

`perf` attributes the whole prefill wall-clock time [HW sweep]. Outside the offloaded path, CPU
matmul is the largest term at 62.6%, and it is one op the entry declines. CPU flash attention is
the second, at 19.2%.

**Conditions:** H96 MAX M9, Qwen2.5-1.5B F16 on llama.cpp b10356, `-p 2048 -ub 512 -r 3`,
`ROCKET_INT8=1`, `ROCKET_FLASH_ATTN=0`, `taskset -c 4-7 -t 4`, governor pinned.

| term | share of prefill wall-clock time |
|---|---:|
| CPU matmul, one tinyBLAS f16 kernel | **62.6%** |
| CPU flash attention, `ggml_compute_forward_flash_attn_ext_tiled` | **19.2%** |
| ggml-rocket host work, `libggml-rocket` user space | 8.0% |
| kernel, the rocket `alloc` / `submit` / `wait` ioctls | 5.3% |
| `lrintf32` and `memcpy`, called by the ggml-rocket quant path | 2.5% |
| ggml's f16 <-> f32 conversion | 1.1% |
| norms + elementwise + copies + every thread barrier | **0.7%** |
| page faults and model load | 0.34% |

> [!NOTE]
> The shares above are wall-clock shares, which `perf` does not report by default. Record with
> `perf record --latency` to get them.

Read as cycles, the ggml-rocket host work is 3.56% and ggml's GEMM is 91.67%. Weighted by
1/parallelism, they are 7.97% and 82.52%. The 2.2x gap on the ggml-rocket side is exactly the
parallelism gap. The ggml GEMM runs on four threads, while the quant, the packs and the scans
are single-threaded.

Norms, elementwise, copies and every thread barrier together are 0.7%, and the scheduler and
graph allocation do not appear at all. Attention is 19.2%. It is a fused CPU kernel, so
declining `FLASH_ATTN_EXT` gives the CPU's tiled implementation rather than a naive path.

The CPU matmul is one op the entry declines. The entry refuses `K >= 6176`, and the model's FFN
down-projection has `K`=8960, so all 28 of them run on the CPU. They are 28 of 196 projections
and 29.4% of the model's projection MACs (`down` is 13.76M of 46.8M MAC a layer a token, tied
with `gate` and `up`).

The control shows that op is essentially the whole 62.6%. With `ROCKET_INT8=0`, every
projection runs on the CPU, and the same kernel is 91.4% of a 736.9 s run over 8 forwards.
There it covers 3.399 MAC-units a layer, where the offloaded configuration's CPU side covers
1.0. At the control's own per-unit rate, the 28 declined GEMMs are 24.8 s of the offloaded
configuration's 50.2 s a forward. Measured in that configuration, they are 31.4 s.

The declined op is therefore 49-63% of the wall-clock time. The 13-point spread is either cache
interference from the interleaved ggml-rocket host work or the two other matmuls the entry does
not take. `perf` cannot separate them, because both are the same symbol.

| per MAC-unit (1 unit = one `gate`/`up`/`down` a layer = 1.973e11 MAC a 512-token forward) | s |
|---|---:|
| four A72 cores, ggml tinyBLAS f16 | **24.8** |
| the NPU offloaded path, converged | **3.86** |
| ratio | **6.4x** |

The 6.4x compares with the 5.4x median of the standalone GEMM comparison above. That
comparison's 2.39x model-weighted figure is an end-to-end cap over a whole GEMM mix, not a
per-MAC rate.

The path's converged `pp2048` speedup is 1.96x: 10.91 t/s against the CPU F16 control's
5.57 t/s. The build, the threads and the governor are the same, and the W8A8 path is the only
difference. The figure is taken outside the calibration regime.

> [!NOTE]
> At `pp512` every offloaded call is a calibration forward. The `pp512` 1.02x / 2.01x pair is
> a 100% calibration measurement, not one of the converged path.

## The W8A8 path against ggml Q8_0 at `pp2048`

Against ggml's Q8_0 CPU kernel at `pp2048`, the W8A8 path is 1.31x [HW sweep, 2026-08-23].
Against the F16 GGUF on the CPU it is 2.37x.

**Conditions:** H96 MAX M9, `rocket` 1.6.0, Qwen2.5-1.5B, llama.cpp b10558, `ggml-rocket` build
of 2026-08-19, measured 2026-08-23. `pp2048` and `-ub` 512, four threads on the A72s, governor
`performance`, one job in flight, three processes a configuration interleaved in one boot.

| configuration | GGUF | `pp2048` t/s (three processes) | mean | ratio |
|---|---|---|---:|---:|
| NPU, W8A8 | F16 | 13.37 / 13.34 / 13.38 | 13.36 | n/a |
| CPU only | Q8_0 | 10.31 / 9.95 / 10.27 | 10.18 | **1.31x** |
| CPU only | F16 | 5.64 | 5.64 | 2.37x |

The F16 ratio is what a drop-in user sees, because the path consumes an F16 GGUF. The Q8_0 ratio
is what a user who already quantizes sees. Quote the two together.

Run-to-run spread at `pp2048` is small: +-0.02 t/s within a process and 3.5% across the three
Q8_0 processes. One process is therefore a measurement at `pp2048`. The path's +-15% spread is
a property of the `pp512` calibration regime.

The largest prefill term on the RK3576 is the `K >= 6176` refusal, not anything inside the
entry. Without the host [K-split](../GLOSSARY.md) that lifts it, the refusal leaves 62.6% of the
prefill wall-clock time on the CPU. Every optimization inside the entry is bounded by the
offloaded path's 21.8%. The two main prefill optimizations are the host K-split and resident
weights, measured separately. The K-split lifts the refusal (§"The host K-split: prefill speed
and perplexity"). Resident weights with the BO pool raise `pp2048` throughput 10.0% over the
per-call path at `-ub` 512, with the K-split off (§"Supplied sums, the resident weight cube and
the BO pool").

### The cost of a host K-split of `ffn_down`

The entry's cost is linear in `K` at fixed (`M`, `N`), with a small per-call intercept. Its
`K`-proportional terms do not depend on how `K` is cut. A whole `K`=8960 `ffn_down` in `n`
chunks therefore costs 1.94`n` + 88.1 ms [HW sweep, 2026-08-11]. The chunk count is nearly
free. Summing independently requantized int8 partials adds no measurable error.

**Conditions:** H96 MAX M9, `rocket` 1.6.0, warm repetitions of
`rocket_matmul_int8_rk3576_perc` at `M`=512 `N`=1536 with the first discarded, governor pinned
to `performance`, measured 2026-08-11.

**Not characterized:** a real activation distribution, and the rotation block size a real split
would also change. The error measurement uses pseudorandom Gaussian operands with no rotation.

| `K` | 1536 | 1792 | 2048 | 2176 | 2304 |
|---|---:|---:|---:|---:|---:|
| ms a call | 17.0 | 19.0 | 20.9 | 23.5 | 24.6 |

The fit is 1.94 + 0.00983·`K` ms, with residuals under 0.6 ms over twelve test points on a 64
step. The `K`-proportional term is split-invariant. These terms are all O(`N*K`) or O(`M*K`)
per call:

- `sumabs`
- The weight cube's `memset`
- `packB`
- `packA`

They sum to the same total however `Ktot` is cut, and the weight BO's `alloc` scales with it
too. The cost is 95.9 ms at `n`=4. A four-chunk run measured 92 ms of wall-clock time and 82 ms
of accounted buckets [HW sweep, H96 MAX M9, rocket 1.6.0, 2026-08-11].

Four chunks (3 x 2304 + 2048) each have their own two-pass bootstrap and frozen per-column
scale, and each partial is dequantized by that scale and summed in f32. Against the exact int32
product over the whole `Ktot`=8960, the composed RMS relative error is 2.2385%. The
single-chunk control gives 2.2448% on chunk 0 alone, a ratio of 0.997. The default path gives
2.234% at `K`=1536 `N`=1536 [HW sweep, H96 MAX M9, one seed, 2026-08-11].

The pure planner, `rocket_matmul_plan_int8_rk3576`, costs no device time. Its map over (`M`,
`K`, `N`) puts the refusal boundary at `N`=1536 at exactly `K`=6176, independent of `M` across
{58, 128, 256, 512, 1024, 2048}. The output tile narrows on the way there. The tile width `Nt`
is 1536 up to `K`=4096, 768 at `K`=4480-4608 and 32 at `K`=5120-6144 [host-computed]. Over the
same range `Mt` falls 128 -> 64 -> 32.

The planner's `Mt`/`Nt` are the entry's tiling. The return value of
`rocket_matmul_plan_int8_rk3576` is `ceil(N/Nt) * ceil(ih/rows)`, N tiles times row tasks. The
profiler in `r76_mm_int8` increments `tiles` once per N tile and `submits` once per row task.
Model-weighted over Qwen2.5-1.5B's six offloaded projections at `M`=512, the plan gives 2.333
tiles and 9.333 submits a call. The profiler measures 2.301 and 9.20, the same 0.986 ratio on
both, which is what a few zero-tile calls (refusals, folded on every exit) give.

At `M`=512 `N`=1536 the plan reads:

| `K` | planned submits |
|---|---:|
| 640 | 1 |
| 896-1280 | 2 |
| 1536-2560 | 4 |
| 2880-4096 | 8 |
| **4480-4608** | **32** |
| 5120-6144 | 768 |

At `K`=4480, `Nt` first collapses 1536 -> 768. A 2 x 4480 cut is therefore the worst one on the
planner's map. Two 4480 chunks plan 64 submits, where four chunks of ~2300 plan 16 at the same
four N tiles. Any cut into the 2880-4096 band costs 8 submits a chunk, and the 1536-2560 band
costs 4 [host-computed, pure planner].

> [!NOTE]
> The raw convolution path's row-window planner refuses `M`=58 `K`=4480 `N`=1536. Its message
> is "no row window fits, the weight slice alone exceeds the CBUF pool at ic=4480 k=1x1". The
> matmul entry tiles with its own planner and accepts the same shape.

### The host K-split: prefill speed and perplexity

At `pp2048` with `-ub` 2048, the host K-split offloads `ffn_down` and raises prefill throughput
2.26-2.37x over the unsplit path [HW sweep 2026-08-11]. Under the default per-tensor activation
scale it also raises Qwen2.5-1.5B's perplexity 1.60x. With the per-row activation scale
(`ROCKET_RK3576_AROW=1`) the split measures 1.0010x the unsplit path (§"Per-row activation
scale"). The split is off by default, and `ROCKET_RK3576_KSPLIT=1` turns it on.

The frontend file `ggml-rocket.cpp` holds each RK3576 W8A8 weight as a list of K-chunks, one
whole W8A8 GEMM per chunk. Each chunk has its own:

- Rotation
- Per-output-channel weight scale
- Two-pass bootstrap
- Frozen per-column output scale

The frontend sums the dequantized f32 partials. The default preference cuts `K`=8960 to
3 x 2304 + 2048.

> [!CAUTION]
> With `ROCKET_RK3576_KSPLIT=1` and the default per-tensor activation scale, Qwen2.5-1.5B's
> perplexity rises 1.60x, and nothing reports it.
> The entry's saturation report fires only above 0.1% of the output, and it stays silent here.
> Software must keep the split off for a model whose perplexity has not been measured with it.

#### Prefill throughput

**Conditions:** H96 MAX M9, Qwen2.5-1.5B F16, `pp2048 -b 2048 -ub 2048 -r 3`, governor pinned to
`performance`, the split against the same build's `ROCKET_RK3576_KSPLIT=0`.

| pass | nosplit t/s | split t/s | ratio |
|---|---:|---:|---:|
| 1 | 11.71 +- 1.59 | 26.42 +- 8.63 | **2.26** |
| 2 | 11.39 +- 1.51 | 27.01 +- 9.22 | **2.37** |

The means understate the converged ratio, because they include calibration. At `-ub 2048` a
forward is one call per projection, so `NCAL`=2 uses the warm-up and the first timed rep for
calibration. The split's calibration is four chunks x two bootstrap passes, where the
control's is one x two.

> [!NOTE]
> The path's 1.96x (§"Prefill time by function") is against a pure-CPU F16 configuration that
> was not re-run at these settings. Do not multiply either ratio above into it.

The `ROCKET_MM_PROFILE` buckets, differenced between the two configurations over their 27 extra
claimed matmuls at `M`=2048, give the split's added cost:

| term | split - nosplit, per `ffn_down` call at `M`=2048 | quartered to `M`=512 |
|---|---:|---:|
| `entry` | 285 ms | 71 |
| `quant+rot` | 289 ms | 72 |
| `calibrate`, per calibration forward | 841 ms | 210 |

The `M`=2048 column is the measurement, and the rest is arithmetic. The quartering assumes both
terms are `M`-proportional, and the entry's term is so only in part: `packA` is O(`M*K`), and
`sumabs` and `packB` are not. The `calibrate` term is the largest of the three. It is 122 s even
without the split, over two forwards, on a run whose converged forward is 77 s.

#### Perplexity

**Conditions:** Qwen2.5-1.5B on wikitext-2, 8 windows of 512, one build and one boot,
`llama-perplexity`.

| configuration | what runs on the NPU | PPL | x CPU | x nosplit |
|---|---|---:|---:|---:|
| `cpu` | nothing (F16 CPU) | 9.7872 | 1.000 | |
| `nosplit` | q/k/v/o/gate/up at `K`=1536, `ffn_down` on the CPU | 9.8969 | 1.011 | 1.000 |
| `machinery` | the same set, but `K`=1536 forced to **512+512+512** | 9.8211 | 1.003 | **0.992** |
| `split` | + `ffn_down` as **3 x 2304 + 2048** | 15.8395 | 1.618 | **1.600** |
| `chunk1024` | + `ffn_down` as **8 x 1024 + 512 + 256** | 18.2188 | 1.861 | **1.841** |

The split machinery adds no error. The `machinery` configuration leaves `ffn_down` on the CPU,
so its offloaded set is identical to `nosplit`. It cuts only the `K`=1536 projections, with the
rotation held fixed. It comes back at 0.992x, slightly better than unsplit, because each
chunk's per-tensor activation scale covers a narrower slice. The 1.600x is what happens when
`ffn_down` reaches the W8A8 path.

The frozen per-column output scale does not cause the 1.600x, in either direction. The entry's
saturation report does not fire on either split configuration. Cutting `CALSAFE` 3.0 -> 1.2
makes the model worse (15.84 -> 17.99), so restoring resolution costs more than it returns.
### Chunk depth and the rotation's block width

Under a per-tensor activation scale, the chunk depth is the K-split's accuracy setting. On
Qwen2.5-1.5B it moves perplexity further than the split's whole 1.60x cost, from 1.459x `nosplit`
at the best depth measured to 2.460x at the worst. Neither the rotation's block width nor the
chunk count orders the results alone, and the per-tensor scale accounts for the effect
(§"Chunk depth under a per-row scale").

**Conditions:** Qwen2.5-1.5B, `ffn_down` at `K`=8960, per-tensor activation scale, eight
perplexity windows. Every test point has the same offloaded set, build and boot, and only
`ROCKET_RK3576_KCHUNK` changes.

| Configuration | `ffn_down` cut as | Chunks | Weighted `H` block width | PPL | x `nosplit` | Windows worse than `split` |
|---|---|---:|---:|---:|---:|---:|
| `k2048` | 4x2048 + 512 + 256 | 6 | 1909 | **14.4401** | 1.459 | 3 of 8 |
| `split` (default cut) | 3x2304 + 2048 | 4 | 666 | 15.8395 | 1.600 | n/a |
| `k6144` | 6144 + 2304 + 512 | 3 | 1499 | 17.9495 | 1.814 | 6 of 8 |
| `chunk1024` | 8x1024 + 512 + 256 | 10 | 973 | 18.2188 | 1.841 | n/a |
| `k1792` | 5x1792 | 5 | 256 | 21.8054 | 2.203 | **8 of 8** |
| `k4096` | 4096 + 4096 + 512 + 256 | 4 | 3781 | 22.0116 | 2.224 | **8 of 8** |
| `k1280` | 7x1280 | 7 | 256 | 24.3497 | 2.460 | **8 of 8** |

The frontend's rotation is block-diagonal `H_b`, with `b` the largest power of two dividing the
chunk's `K` (`rk_hadamard_rotate_blocks`). A chunk depth therefore sets how far an outlier is
spread. A chunk of `K`=2048 gets a full 2048-wide fast Walsh-Hadamard transform (FWHT), and one of
`K`=1792 gets `H_256` seven times over.

At a depth that is not a power of two, the chunk count and the rotation width move together.
Neither orders the results alone:

- At a fixed `b`=256, five chunks score 21.81 against 24.35 for seven. The count term is real and
  much smaller than the spread in the table.
- `k4096` has the widest rotation reachable, 91% of `K` under a full `H_4096`. Every chunk is a
  power of two, and the chunk count is the default cut's. It scores 22.01, level with the
  narrowest depth measured.
- `k6144` has the fewest chunks and twice the default cut's weighted width, and it is 2.11 PPL
  worse than the default cut.

The pure power-of-two depths give every chunk a full-width FWHT, so the rotation construction is
constant across them and only the depth changes. The depth has an interior optimum at 2048:

| Pure power-of-two depth | Cut | Chunks | PPL |
|---:|---|---:|---:|
| 1024 | 8x1024 + 512 + 256 | 10 | 18.2188 |
| **2048** | 4x2048 + 512 + 256 | 6 | **14.4401** |
| 4096 | 2x4096 + 512 + 256 | 4 | 22.0116 |

Every test point above uses a per-tensor activation scale, and that scale is the whole of the
effect. With a per-row scale, the same three chunkings measure within 0.04 PPL of each other and
of `nosplit` (§"Chunk depth under a per-row scale"). The optimum belongs to the scale's
granularity, not to the depth.

At eight windows, only `k1792` and `k1280` separate cleanly from the default cut. The `k2048`
configuration's 1.459x is the best test point measured, and not an established better
configuration.

### Per-row activation scale

The K-split's accuracy term is the per-tensor input scale. With a per-row activation scale
(`ROCKET_RK3576_AROW=1`), the split path brings `ffn_down` to `nosplit` quality on Qwen2.5-1.5B
[HW sweep 2026-08-11]. It measures 9.9071, against the same binary's default-path 15.9894. The
result is 1.0010x of the `nosplit` control's 9.8969, and 1.012x of the CPU F16 configuration. It
reproduces to four decimals across builds and boots (9.9348 twice, 9.9071 on the other build).

**Conditions:** H96 MAX M9, `rocket` 1.6.0, kernel 7.1.7. Qwen2.5-1.5B, `ffn_down` at the default
cut, eight perplexity windows.

The per-row configuration keeps `ffn_down` on the NPU throughout. It runs at 67.78 s per pass, the
split configuration's own speed, against `nosplit`'s 80.29.

The per-row scale needs no library change. The library entry takes A already quantized, and its
`scale` parameter is the output's, so a per-row activation scale never crosses the ioctl. The
frontend normalizes each row to +-127 before the submit and applies the row factor in the host
dequantize.

That binary's default path measures 15.9894 rather than 15.8395 because of an fp32 re-association
in its dequantize. The library keeps the per-tensor dequantize's original association, and the
per-row scale is a separate branch.

The per-axis input-scale result elsewhere in this note is better on every per-GEMM norm, and 38x
worse end to end on one model of two. It is measured on `K`=1536 projections under a different
output quantization, and this result on `ffn_down`. Neither result bounds the other, and neither
bounds a second model.

The K-split itself is fast and exact. The operation it moves to the NPU, `ffn_down`, needs an
activation quantization that the default per-tensor scale does not give it.

> [!CAUTION]
> With the K-split on and a per-tensor activation scale, the model's output degrades. Perplexity
> is 15.8395 against 9.9348 on Qwen2.5-1.5B. Software must turn `ROCKET_RK3576_AROW` on whenever
> `ROCKET_RK3576_KSPLIT` is on.

### Chunk depth under a per-row scale

With `ROCKET_RK3576_AROW=1`, the depth ladder's 9.9 PPL spread falls to 0.04, and every chunking
sits inside one standard error of `nosplit`. On this path `ROCKET_RK3576_KCHUNK` sets cost, not
accuracy [HW sweep 2026-08-11].

**Conditions:** H96 MAX M9, `rocket` 1.6.0, kernel 7.1.7. Qwen2.5-1.5B, eight perplexity windows,
`nosplit` control 9.8969. The three test points hold the chunking fixed against the per-tensor
table above, and change only the activation scale's granularity. The paired dNLL is the per-window
difference in mean negative log-likelihood (NLL), in nats.

| Cut | Chunks | Per-tensor scale | Per-row scale | Paired dNLL against `nosplit` | Windows worse |
|---|---:|---:|---:|---|---:|
| 3x2304 + 2048 (default) | 4 | 15.8395 | **9.9348** | +0.0038 +- 0.0048 | 5 of 8 |
| 4x2048 + 512 + 256 | 6 | 14.4401 | **9.8944** | -0.0003 +- 0.0045 | 6 of 8 |
| 7x1280 | 7 | 24.3497 | **9.9012** | +0.0004 +- 0.0045 | 5 of 8 |

The term the chunk depth trades is the per-tensor scale's reach across rows. A chunk's divisor is
a `max` over `M*K` rotated values, so one outlier row sets the resolution for every other row of
that chunk. The cross-row reach accounts for essentially all of the effect. The number of
independently requantized partials, an output-side count that no input scale touches, contributes
at most 0.04 PPL over 4 to 7 chunks.

The interior optimum at 2048 belongs to the per-tensor scale and does not survive the per-row
scale. Choose a chunk size by the entry's cost law and its stall `K` alone
([../chips/rk3576.md](../chips/rk3576.md#chunk-depths-for-the-k-split) §"Chunk depths for the
K-split" and §"The weight-phase rule").

**Not characterized:** a chunking far outside 4-7 pieces, and a `K` other than 8960.

### SmolLM2-1.7B

On SmolLM2-1.7B the per-tensor split costs 6.77x perplexity, and the per-row scale restores
0.9992x `nosplit` [HW sweep 2026-08-11]. The model's `ffn_down` is `K`=8192, which the greedy
chunker cuts into 3x2304 + 1024 + 256. The cut has five chunks, with a different tail from
Qwen2.5-1.5B's four.

**Conditions:** H96 MAX M9, `rocket` 1.6.0, kernel 7.1.7. SmolLM2-1.7B, eight perplexity windows,
the configurations of §"Per-row activation scale".

| Configuration | PPL | x `nosplit` | Paired dNLL | Windows worse |
|---|---:|---:|---|---:|
| `nosplit` | 10.3436 | n/a | n/a | n/a |
| `split` | **70.0420** | **6.77** | +1.9127 +- 0.6832 | 8 of 8 |
| `split` + `AROW` | **10.3354** | **0.9992** | -0.0008 +- 0.0043 | **3 of 8** |

The per-tensor split hurts this model 4.2x harder than it hurts Qwen2.5-1.5B: 6.77x against
1.600x. The entry's saturation report fires at 15.417% of the output on `blk.7.ffn_down`, where
Qwen2.5-1.5B's worst configuration reaches 0.225%. With the per-row scale the model is
indistinguishable from `nosplit`, with a paired difference below zero and worse on only 3 of 8
windows.

The per-row fix is therefore not specific to Qwen2.5-1.5B, and the per-axis input-scale result
(§"Per-row activation scale") does not bound it. On both models, the K-split's accuracy cost is
the per-tensor activation scale, and nothing else measured here.

### Llama-3.2-1B-Instruct

The per-tensor split's damage depends on the model's weights rather than on the cut. The
Llama-3.2-1B-Instruct model has SmolLM2-1.7B's `K` set exactly, and so its `K`=8192 cut. At eight
windows it measures 2.0100x `nosplit`, 1.26x from Qwen2.5-1.5B's 1.600x and 3.37x from
SmolLM2-1.7B's 6.772x [HW sweep 2026-08-20].

**Conditions:** H96 MAX M9, `rocket` 1.6.0, kernel 7.1.7. Llama-3.2-1B-Instruct, `n_embd` 2048 and
`n_ff` 8192. Wikitext-2 test at `--chunks 64`, `-ub 512`, one process a test point.

| Configuration | What runs on the NPU | PPL, 64 windows | x `nosplit` |
|---|---|---:|---:|
| `nosplit` | q/k/v/o/gate/up at `K`=2048, `ffn_down` on the CPU | 15.3461 | 1.000 |
| `ctl` = `split` + `AROW` | + `ffn_down` cut, per-row activation scale | 15.4192 | 1.0048 |
| `ncal1` | `ctl` at `NCAL`=1 | 15.5343 | 1.0123 |
| `split` | + `ffn_down` cut, per-tensor scale | **37.2952** | **2.4303** |

The whole-offload cost gives the same answer from an independent contrast. The `ctl` configuration
against `nosplit` is +0.476% (paired +0.0048 +/- 0.0021 of mean NLL). The figure sits with
Qwen2.5-1.5B's +0.567% and not with SmolLM2-1.7B's -0.079%.

The split's damage grows with depth, and the whole-offload cost varies over the same three depths:

| Windows | `split` x `nosplit` | Whole-offload cost, `ctl` against `nosplit` |
|---:|---:|---:|
| 8 | 2.0100x | +0.863% |
| 29 | 2.3434x | +0.123% |
| 64 | 2.4303x | +0.476% |

The 8-window ratios recorded for the other two models are therefore lower bounds on their own
64-window values. The size of the difference between the models at that depth is not measured, and
its direction holds at matched depth.

The `NCAL`=1 setting has a small, positive cost in the converged windows. Pooled by inverse
variance over the three models, it is +0.00564 +/- 0.00199 nats, +0.566% of perplexity, and 2.83
standard errors (se) from zero. On this model alone, `ncal1` sits +0.0074 +/- 0.0025 se above
`ctl` over windows 3-64, 3.0 se from zero.

**Not characterized:** another corpus or `-ub`, prompts after the first, decode, and a `K` set
other than {2048, 8192}. This model is instruction-tuned where SmolLM2-1.7B is a base checkpoint,
and these test points cannot separate that from model family.

### Perplexity measurement resolution

Perplexity on the W8A8 path is deterministic to four decimals, so a figure's uncertainty is the
sample of windows it covers. The paired per-window difference has a standard deviation (sd) of
0.01734 nats over 64 windows [HW sweep 2026-08-13]. The resolution at 2 se is `2*sd/sqrt(n)`.

**Conditions:** H96 MAX M9, `rocket` 1.6.0, kernel 7.1.7. Qwen2.5-1.5B, the split on. The text is
`wiki-test-full.txt`, the non-blank lines of wikitext-2-raw's `wiki.test.raw`. It holds 298938
tokens under Qwen2.5's tokenizer, 583 windows at `n_ctx`=512.

| Windows | 2-se resolution (nats) | 2-se resolution (PPL) | Time, Qwen2.5-1.5B, split on |
|---:|---:|---:|---:|
| 29 | 0.0064 | 0.65% | 11 min |
| **64** | **0.0043** | **0.43%** | **21 min** |
| 128 | 0.0031 | 0.31% | 42 min |
| 256 | 0.0022 | 0.22% | 84 min |
| 583 (the whole text) | 0.0014 | 0.14% | ~3.2 h |

An effect of size *d* needs `(2*sd/d)^2` windows to reach 2 se. The `AROW`-scoping recovery of
0.00127 (§"Per-row scale on the split weights only") needs 746, against the 583 the whole of
wikitext-2 test gives. The recovery cannot be established against zero on this corpus at any
depth, and deciding its sign needs a different corpus, not a longer run.

### `AROW` without the K-split

The `ROCKET_RK3576_AROW` setting applies to every offloaded GEMM [HW sweep 2026-08-13]. With the
split off, it is a net loss of 0.415% of perplexity. The frontend reads `ROCKET_RK3576_AROW` once
per context, and hands it to `rk76_quant_act` and `rk76_dequant` on every offloaded GEMM. The
K-split decides only which GEMMs are offloaded.

> [!NOTE]
> `ROCKET_RK3576_AROW` stays in effect with `ROCKET_RK3576_KSPLIT=0`. To measure the split alone,
> turn `AROW` off as well.

At `-ub` 512, Qwen2.5-1.5B offloads 197 GEMM sites under `KSPLIT`=1. Of those, 169 are taken whole
at K=1536 in one chunk, and 28 are cut at K=8960 into `2304+2304+2304+2048`, one per layer. Under
`KSPLIT`=0 the entry's `K >= 6176` bound refuses those same 28, and they run on the CPU. The two
`nosplit` test points below therefore differ on exactly the 169.

**Conditions:** H96 MAX M9, `rocket` 1.6.0, kernel 7.1.7. Qwen2.5-1.5B, one build and one boot,
paired window by window at 29 and 64 windows. A perplexity level is comparable only within a
depth, and the paired difference is the quantity.

| Configuration | `KSPLIT` | `AROW` | PPL @29 | PPL @64 | Against the row above, @64 |
|---|---:|---:|---:|---:|---:|
| `nosplit_a0` | 0 | 0 | 8.5236 | 9.3653 | n/a |
| `nosplit` | 0 | 1 | 8.5723 | 9.4042 | **+0.415%**, paired +0.00415 +/- 0.00208 (1.99 se) |
| `ctl` | 1 | 1 | 8.5745 | 9.4184 | **+0.151%**, paired +0.00151 +/- 0.00197 (0.77 se) |

Only the aggregate is established. The whole offload feature, `ctl` against `nosplit_a0` with both
the setting and the split, costs +0.567%, paired +0.00565 +/- 0.00204 (2.77 se). It is stable
across depths (+0.597% at 29). Neither of the two terms it divides into reaches 2 se on its own.
The decomposition, and any ratio between the terms, is therefore not established.

The sign of the setting's effect depends on the split, and the effect is large rather than
statistical. With the split on, the per-row scale is required (§"Per-row activation scale"). With
the split off, it is a loss on every site that never needed it. Per-axis input scales measured end
to end elsewhere in this note go the same direction, at a different interface.

The 8-window pair `nosplit` 9.8969 against `arow` 9.9348 (+0.38%) is the setting and the split
together, not the split alone. It is 15x the split's own 8-window figure for that reason.

### Per-row scale on the split weights only

With `ROCKET_RK3576_AROW=2`, the per-row activation scale applies only where the K-split cuts the
weight (`w.ch.size() > 1`). Against `AROW`=1 it recovers at most 0.56% of perplexity at 2 se
[HW sweep 2026-08-13]. That bound contains the setting's whole 0.415% cost with the split off,
measured at the same 64 windows. This corpus cannot separate the recovery from zero.

The mode needs no second activation quantization. The decision is a property of the weight, and
every chunk in one call belongs to one weight. The calibration forwards take the same flag, so the
frozen scales match the mode. The chunk map is unchanged (169 x 1 + 28 x 4).

**Conditions:** H96 MAX M9, `rocket` 1.6.0, kernel 7.1.7. Qwen2.5-1.5B, `KSPLIT`=1, 29 and 64
windows, one `-ub`, a first prompt only.

| Configuration | Windows | `KSPLIT` | `AROW` | PPL | Against `ctl` |
|---|---:|---:|---:|---:|---:|
| `ctl2` | 29 | 1 | 1 (all 197) | **8.5745** | n/a (reproduces `ctl` byte-for-byte on all 29) |
| `arow2` | 29 | 1 | 2 (28 only) | **8.5690** | **-0.064%**, paired -0.0006 +/- 0.0034 (0.19 se) |
| `ctl_L` | **64** | 1 | 1 (all 197) | **9.4184** | n/a (reproduces `ctl` byte-for-byte on its first 29) |
| `arow2_L` | **64** | 1 | 2 (28 only) | **9.4064** | **-0.127%**, paired **-0.00127 +/- 0.00217** (0.59 se) |

At 64 windows the 2-se interval on the paired difference is [-0.0056, +0.0031]. An additive
decomposition predicts a recovery of the setting's own cost with the split off at the same depth,
a paired difference of -0.00415 (§"`AROW` without the K-split"). The interval contains that
prediction and zero [host-computed].
The prediction sits 1.33 se from the point estimate, before its own +/-0.00208, and zero sits
0.59 se from it. The test therefore establishes neither additivity nor its failure. The point
estimate, a recovery of 0.127%, is about a third of the setting's 0.415% cost with the split off.

Whether the setting's cost with the split off transfers to the scoped mode is not established.
Isolating that cost requires turning the split off, which moves the 28 `ffn_down` GEMMs to the
CPU in f32. The scoped mode does not run in that configuration. The offloaded sites are the same
169 by readout.

The four `(KSPLIT, AROW)` combinations do not compose. With `KSPLIT`=1 and `AROW`=0, perplexity is
15.84, because that combination removes the per-row scale from the split sites too. The scoped
mode is a fifth configuration and cannot be interpolated from the four.

The accuracy case for the scoped mode is bounded above by 0.56% at 2 se [host-computed]. It is
not argued from the setting's cost with the split off.

**Not characterized:** other models, prompts after the first, and decode. Whether the scoped mode
gains anything at all is not established, since the recovery is 0.59 se from zero.

### `AROW` cost in the per-call buckets

`AROW` costs 8.5% of the dequantize and nothing else, which is ~0.5% of a forward [HW sweep
2026-08-11]. The net is +1.60 ms a call at 193 calls a forward, +0.31 s against a converged
forward of 62-67 s on the same processes.

**Conditions:** H96 MAX M9, `rocket` 1.6.0, kernel 7.1.7. Qwen2.5-1.5B, `pp2048 -ub 2048`,
governor pinned to `performance` under `taskset -c 4-7`, three interleaved processes a
configuration. Each process makes 772 calls, of which 386 are calibration forwards. The table
gives per-call ms, the bucket total over `calls`.

| Bucket | `split` mean | `arow` mean | Delta | 9 pairings | Control-to-control spread |
|---|---:|---:|---:|---|---:|
| `valloc` | 2.23 | 2.23 | -0.02% | -0.41 .. +0.41 | 0.35% |
| `quant+rot` | 82.62 | 83.45 | +1.00% | -0.70 .. +3.15 | **0.82%** |
| `calibrate` | 220.33 | 224.12 | +1.72% | -0.73 .. +5.11 | 0.63% |
| `entry` | 61.68 | 61.45 | -0.37% | -9.54 .. +11.33 | **9.38%** |
| `dequant` | 9.06 | 9.83 | **+8.50%** | +8.10 .. +8.92 | **0.57%** |
| `quant+rot` + `dequant` | 91.69 | 93.29 | **+1.75%** | +0.18 .. +3.72 | 0.80% |

The whole cost is in `dequant`, at +8.50%. Its nine pairings fall between +8.10 and +8.92, about
15x that bucket's own control spread. The cost is the per-row branch's one extra f32 multiply per
output element per chunk, which the per-tensor branch hoists into `f[n]`.

The per-tensor branch of `rk76_quant_act` writes `rot` in the row loop, then makes a second flat
pass over `rot[M*K]` to quantize. That pass re-reads 73.4 MB per tensor per call at `M`=2048
`K`=8960, which is 2.05 GB a forward. The per-row branch never makes it, because it quantizes each
row while its 9.2 KB is still in a 32 KiB L1d. Without the pass, `quant+rot` moves +1.00%, under
0.6 ms of an 82.6 ms/call bucket, with pairings spanning zero. The loop is therefore not
bandwidth-bound. The likely term in `quant+rot` is `rk76_rotate`, which is unmeasured
[hypothesis].

### Resolution of `pp2048` wall-clock time

A `llama-bench` `pp2048 -ub 2048` comparison cannot measure `AROW`'s cost [HW sweep 2026-08-11].
The control configuration's process-to-process spread is larger than the effect. Over three
identical `split` processes, the spread measured 36% in one run and 4.1% in another. Repetitions
inside one process agree to under 1%, so the state that sets the spread is fixed at process start.

**Conditions:** H96 MAX M9, `rocket` 1.6.0, kernel 7.1.7. Qwen2.5-1.5B, governor pinned to
`performance`, `taskset -c 4-7`, thermals at 47-51 C.

In the 4.1% run, `AROW` measures -1.8%, the sign the per-call buckets give. The control spread is
2.2x that effect, so wall-clock time agrees with the buckets and cannot resolve the effect.

The likely cause of the 36% spread is page-cache contention between a 3.1 GB model mapping and a
second 3.4 GB model file [hypothesis]. The board has 7.9 GB of RAM, and the 36% run fell within
the hour in which the second model was written.

> [!NOTE]
> A `pp2048` wall-clock comparison on this path needs at least three processes per configuration.
> To measure a term inside the W8A8 handler, difference the frontend profiler's `quant+rot` and
> `dequant` buckets. Those buckets are per call and hold to under 1%.

The K-split's recorded 2.26-2.37x was taken over two process pairs, fewer than that.

### Calibration cost in the W8A8 frontend

Calibration is 58.6% of the frontend's accounted time, and with the column-outer scan 45% of it is
one loop nest [HW sweep 2026-08-11]. The `calibrate` bucket is the largest the RK3576 W8A8
frontend accounts for, at 171.3 s a process against 47.5 s of `entry`.

**Conditions:** H96 MAX M9, Qwen2.5-1.5B `pp2048 -ub 2048`, 772 calls a process of which 386 are
calibration forwards, three processes, measured 2026-08-11. The `ggml-rocket` build scans
column-outer, and predates the row-major scan (§"The calibration scan's loop form"), the supplied
weight sums and the resident weight cube. The `calscan` bucket is a subset of `calibrate`: the
bootstrap's per-column maximum scan alone. The profiler prints the split directly.

| Half | Per process | Per call | Control-to-control spread over 3 processes |
|---|---:|---:|---:|
| `calscan`, the per-column max scan | 78.9 s | 102.25 ms | **10.37%** |
| The rest, two device GEMMs a chunk plus the `N`-wide loops | 92.4 s | 119.71 ms | **0.96%** |

The device half is the two extra passes and nothing else. A calibrating call takes 119.71 ms
there, against 59.55 ms of `entry` for a converged call, which is 2.01x. A bootstrap pass costs
what a converged pass costs, and the calibration regime is two of them.

The variance of this path's largest bucket is in the scan, a host memory-access term, and not in
the device half.

### The calibration scan's loop form

The scan's cost is entirely its loop form, and two bit-identical rewrites take it to 1.0% of
calibration [HW sweep 2026-08-11]. The function `rk76_bootstrap_pass` takes each column's maximum.
Written column-outermost, as `for n { for m { C8[m*N + n] } }`, it reads a row-major `M`×`N` int8
output at a stride of `N` bytes. Written `m`-outermost against an `N`-wide running max, it reads
the same output once, sequentially.

The arithmetic is identical. A column saturates exactly at a largest |code| of 127, so the flag
falls out of the maximum. A `llama-perplexity --chunks 4` run returns PPL 12.8615 +/- 1.17731 to
every printed digit under all three forms.

**Conditions:** Qwen2.5-1.5B `pp2048 -ub 2048`, governor pinned, three interleaved processes a
configuration. The table gives per-call ms.

| Bucket, per call | Column-outer | Row-major | Delta | 9 pairings | Own control spread |
|---|---:|---:|---:|---|---:|
| `calscan` | 102.25 | 10.72 | **-89.52%** | -90.07 .. -89.02 | 10.37% |
| `calibrate` | 221.95 | 128.73 | **-42.00%** | -43.86 .. -40.41 | 5.20% |
| `entry` | 59.55 | 58.39 | -1.95% | -4.26 .. +1.38 | 2.54% |
| `quant+rot` | 78.99 | 79.04 | +0.06% | -3.95 .. +2.61 | 2.75% |
| `dequant` | 9.09 | 9.08 | -0.06% | -0.27 .. +0.16 | 0.14% |
| `valloc` | 2.22 | 2.21 | -0.10% | -0.29 .. +0.12 | 0.18% |

In wall-clock time the row-major form is 1.42x on the forward a user meets. At `NCAL`=2, the
warm-up process's forward and timed rep 1 are the calibrating ones. Rep 1 goes 140.02 -> 98.59 s
(-41.43 s). Reps 2 and 3 move -2.9% and -2.8%, inside the process-to-process drift. A
single-prompt prefill is 100% calibration regime, so 1.42x is what a first prompt gets. A
converged prefill is unchanged.

The row-major form leaves one term, the scan's scalar rate, and the byte form removes it. With an
`int` accumulator, the row-major form measures 1.30 ns an element on the device, a scalar rate.
The conditional store `if (v > mx[n]) mx[n] = v;` does not vectorize. Written unconditionally, the
loop vectorizes. Held in a `uint8_t` instead of an `int`, it vectorizes 16 elements at a time
instead of 4. A standalone microbenchmark on the board measures all four forms.

**Conditions:** standalone microbenchmark on the board, `M`=2048, best of 3, at -O2 and -O3.

| ns/element, -O2 / -O3 | `N`=256 | `N`=1536 | `N`=8960 |
|---|---:|---:|---:|
| column-outer | 5.69 / 5.34 | **17.74 / 16.25** | 12.33 / 12.44 |
| row-major, conditional store, `int` | 1.76 / 1.33 | 1.73 / 1.38 | 1.74 / 1.32 |
| row-major, unconditional store, `int` | 0.41 / 0.35 | 0.41 / 0.52 | 0.43 / 0.35 |
| row-major, unconditional store, `uint8_t` (default) | **0.11 / 0.11** | 0.13 / 0.20 | 0.13 / 0.14 |

The byte accumulator's gain is its lane count [HW sweep, standalone, -O2 and -O3, 2026-08-11]. The
gain is flat at 2.4-3.7x over the unconditional `int` form (`rowmax_uncond`) at every `N`. It is
largest at `N`=256, where the `int` accumulator is 1 KB.

The narrowing is exact rather than an approximation [HW sweep, standalone, -O2 and -O3,
2026-08-11]. Over an int8 output |code| is 0-128, which a byte holds including the `-128` rail. A
code of 128 >= 127 saturates as before.

The byte form is the default. The flag `ROCKET_RK3576_CALSCAN` selects the form: 0 is
column-outer, 1 is row-major `int` and 2 is row-major byte, the default.

**Conditions:** the byte form against `CALSCAN=1`, three interleaved processes a configuration,
governor pinned.

| Bucket, per call | `int` accumulator | Byte accumulator | Delta | 9 pairings | Own control spread |
|---|---:|---:|---:|---|---:|
| `calscan` | 12.41 | **1.18** | **-90.51%** | -90.50 .. -90.52 | 0.05% |
| `calibrate` | 133.36 | 120.88 | -9.36% | -11.15 .. -6.97 | 1.72% |
| `cal_nonscan` (the device half) | 120.95 | 119.70 | -1.03% | -3.09 .. +1.68 | 1.90% |
| `entry` | 60.38 | 59.43 | -1.57% | -6.21 .. +3.66 | 4.60% |
| `quant+rot` | 79.17 | 79.76 | +0.74% | -0.62 .. +3.51 | 4.09% |
| `dequant` | 9.14 | 9.09 | -0.58% | -0.69 .. -0.47 | 0.16% |
| `valloc` | 2.21 | 2.21 | +0.10% | -0.18 .. +0.35 | 0.18% |

The measured 10.5x falls inside the standalone's 9.4-13.4x for the same two forms (1.32-1.74 ->
0.13-0.14 ns an element), so the device reaches the emitted code's rate. The scan is 1.0% of
`calibrate`, against 45% in the column-outer form. In wall-clock time the gain is on the
calibrating rep alone. Rep 1 goes 91.28 -> 86.34 s (1.057x), and reps 2-3 move -0.87% and -0.94%
[HW sweep 2026-08-11].

> [!NOTE]
> A bucket's level can move between two builds in which its own source is unchanged. The
> `calscan` bucket moved 15.8% (10.72 -> 12.41) between the two tables above. Read a bucket's level from the run that produced
> the delta, never from a stored one.

The column-outer scan's cost is L1d set conflict [HW sweep, H96 MAX M9, 2026-08-11]. At 64-byte
lines and 128 sets, stride 1536 gives `gcd(24, 128)`=8, and so 16 of 128 sets, against 32 for both
8960 and 256. The stride of 1536 is accordingly the more expensive per element, at 1.32x `N`=8960.
The row-major form is flat across all three `N`, as expected of a sequential pass.
### Host predictors for bootstrap pass one

Bootstrap pass one's estimate of a column's maximum is never more than 1.488x below the true
value, and the default `BOOTMARGIN`=1.5 covers that error. The bound, rather than the
estimate's accuracy, is what keeps pass two from saturating, so a host predictor cannot
replace pass one. The two host predictors measured have no bound on their underestimate. At
their log-mean fits they saturate pass two on 12.35% and 19.72% of columns
[HW sweep 2026-08-11].

**Conditions:** H96 MAX M9, the W8A8 frontend on Qwen2.5-1.5B at `M`=2048 and `-ub` 2048.
Every column of all 193 offloaded weights over both calibration forwards, 1.50M columns,
scored against the device's own answer. Measured 2026-08-11.

**Not characterized:** perplexity, another model, another `M`, and decode.

A calibrating W8A8 call runs two whole GEMMs, where a converged call runs one. Pass one runs
only to set pass two's scale. It reads back at `cbound` = 127/(128·Σ|qB|+1), the analytic
accumulator bound. Pass two reads back at `sc2` = 127/(est·`BOOTMARGIN`), where est is pass
one's estimate.

`ROCKET_RK3576_CALMAP=<file>` writes one row per column per calibration forward. Each row
holds these fields:

- The analytic bound
- Pass one's estimate
- Pass two's refined estimate
- The saturation flag
- Both operands' second moments

By absolute error, the second-moment predictor is close to pass one. Its median error is 23.5%
against pass one's 11.4%, and it is better than pass one in the tail:

| over 1.50M columns | median | p99 | max |
|---|---:|---:|---:|
| analytic bound's overshoot `abound/est2` | **61.8x** | 1075x | 1.08e4x |
| absolute error of `est1` (pass one, a device GEMM) | 11.4% | 747% | 8367% |
| absolute error of `z·sqrt(Kc)·rms(qA)·rms(qB[n])`, one global `z` | 23.5% | 142% | 794% |
| absolute error of `abound/D`, one global `D` | 25.5% | 1531% | 1.62e4x |

Absolute error does not predict the effect on pass two, whose tolerance is asymmetric. An
estimate above the true maximum costs pass two readback resolution and nothing else. An
estimate more than `BOOTMARGIN`=1.5 below it saturates pass two. The frontend then keeps
`est·1.5`, an underestimate of the column maximum, as the answer. Scored that way, with
`z·G0` the second-moment predictor above, the three separate completely:

| replacement for pass one | columns saturating pass two | worst underestimate | pass two's readback, p50 |
|---|---:|---:|---:|
| **pass one at the defaults** | **0.000%** | **1.488x** | 84 codes |
| `z·G0`, `z` at the log-mean | 12.35% | 342.6x | 84 |
| `z·G0`, `z` set to clear 99.9% of columns | 0.043% | 60.6x | 14.9 |
| `z·G0`, `z` set to clear every column | 0.000% | 1x | **0.245** |
| `abound/D`, `D` at the log-mean | 19.72% | 28.9x | 90.3 |
| `abound/D`, `D` set to clear every column | 0.000% | 1x | 3.12 |

Pass one's error is bounded by construction, so `BOOTMARGIN`=1.5 is structural rather than
fitted. Pass one reads back a real product at a scale that is a proven upper bound. Its
readback maximum is a small integer: median 2 codes, p90 3, p99 5, and 21.5% of columns at
the one-code minimum. Its whole error is that integer's rounding, and 1.5 is exactly the
constant that covers it.

A host predictor has no such minimum, so making it safe on the worst column costs every other
column its resolution. The max-safe second-moment fit puts 99.98% of columns under 8 readback
codes, against 0.83% with pass one at the defaults.

> [!CAUTION]
> A host predictor of the column maximum in place of pass one saturates pass two wherever it
> underestimates by more than `BOOTMARGIN`. The frontend then keeps the underestimate as the
> column maximum, and a host predictor's underestimate has no bound. Software must compute
> pass one's estimate with a device GEMM.

Each K chunk of a split GEMM has its own column maxima, because each chunk's accumulator is
over its own contraction. Across the four chunks of one split `ffn_down`, the median column's
`max_c/min_c` colmax ratio is 1.79x (p10 1.30, p90 11.0, p99 66.3) [HW sweep 2026-08-11].

> [!CAUTION]
> A K chunk calibrated from a neighbor chunk's estimate takes its column maximum from another
> contraction. Software must calibrate each chunk from its own passes.

The second calibration forward raises the stored colmax on 49.9% of columns. It raises 2.03% by
more than 1.5x, and 0.061% by more than 3.0x, which is past the `CALSAFE`=3.0 headroom
[HW sweep 2026-08-11]. The 0.061% is 87x the [row cut](../GLOSSARY.md)'s exposure (next section), and it
is a count of columns. A column that loses coverage clips only its top accumulator rows, and
the perplexity measurement of `NCAL`=1 below does not resolve the loss. These figures bound
what `NCAL`=1 risks, and do not measure its cost.

### A row cut for pass one

Running pass one over `M'` of the `M`=2048 rows cuts the `calibrate` bucket by 28.2% at
`M'`=512 and by 36.3% at `M'`=128. It leaves 0.0007% and 0.0060% of columns under their
full-`M` colmax by more than `CALSAFE`=3.0 [HW sweep 2026-08-11]. `ROCKET_RK3576_CALROWS` caps
pass one's rows, and pass two always runs at the full `M`.

**Conditions:** H96 MAX M9, Qwen2.5-1.5B under `llama-bench -p 2048 -ub 2048`, 750080 columns
scored against the full-`M` run, one process per test point with `ROCKET_RK3576_CALMAP` on,
2026-08-11. The buckets exclude the readout's dump but not its cache effects, so the bucket
figures are a range, not a precise cost.

**Not characterized:** perplexity (next section), decode, and another model.

Pass one's estimate is a maximum over `M` accumulator rows. A row cut keeps the property the
host predictors lack. It is still a readback of a real product at a proven upper bound, with
the same quantum as its minimum. It undershoots only by the gap between the maximum over `M'`
rows and the maximum over `M`. The activations are deterministic, so the full-`M` run's stored
colmax is the reference the row cut reproduces.

> [!NOTE]
> A row cut saturates pass two exactly on the columns where pass one came up short. A
> score over the non-saturating columns alone misses those errors. Score a row cut on the
> stored colmax, over all columns.

| against the full-`M` run, 750080 columns | `M'`=512 | `M'`=128 |
|---|---:|---:|
| pass two saturates, of (column, forward) pairs | 8.04% | 19.82% |
| median stored colmax, row cut/reference | 0.9911 | 0.9792 |
| columns under by more than 1.5x | 0.264% | 1.384% |
| columns under by more than `CALSAFE`=3.0 | **0.0007%** (5) | **0.0060%** (45) |
| worst underestimate | 6.094x | 6.094x |
| `calibrate` bucket | **-28.2%** | **-36.3%** |

At the full `M`, pass two saturates on 0.260% of columns. The median column loses 0.9-2.1% of
its colmax, which `CALSAFE`=3.0 absorbs entirely. The worst column, at 6.094x under, is beyond
it. The perplexity cost of a row cut does not come from these columns of the
frozen scale. It comes from the output of the calibrating forwards, which the table does not
score (next section).

The saving includes the `alloc` and `stamp` buckets. Both scale with the row count, because
the submit count does (31360 -> 16660 at `M'`=128), and `stamp` alone moves -48.9%.

### Row-cut cost and `BOOTMARGIN`

A row cut degrades the output of the calibrating forwards, and raising `BOOTMARGIN` with
it restores that output [HW sweep 2026-08-12]. The frozen scale is not affected. Measured by
perplexity, cutting pass one's rows puts all of the cost on windows 1 and 2, the `NCAL`=2
calibrating forwards. Windows 3-8, the only ones the frozen scale governs, stay inside
+/-0.042 nats, with both signs at every row count tried.

**Conditions:** H96 MAX M9, Qwen2.5-1.5B under `llama-perplexity -ub 512` with `--chunks 8`, 8
windows of one text, so `M`=512 and `CALROWS`=128 is `M`/4. `AROW`=1, K-split on, 2026-08-12.
The [clip fractions](../GLOSSARY.md) in this section (0.26, 8.04 and 19.82%) are `M`=2048
figures from `llama-bench -p 2048 -ub 2048`, where `M`/4 is `CALROWS`=512.

**Not characterized:** another model, another `-ub`, and decode. The next subsection reads
more windows and two more models.

A calibrating call returns pass two's output, at `sc2` = 127/(`est1`·`BOOTMARGIN`). A
row-cut `est1` is low, so `sc2` is high and pass two clips. It clips 8.04% of columns at
`M'`=`M`/4 and 19.82% at `M`/16, against 0.26% at the defaults. The frontend's own comment,
that this output "costs resolution and nothing else", holds only while the output does not
saturate. Raising `BOOTMARGIN` with the row cut restores it, and at full rows the raise costs
nothing:

| Qwen2.5-1.5B, `--chunks 8`, `AROW`=1, K-split on | `BOOTMARGIN` 1.5 | 2.25 | 3.0 |
|---|---:|---:|---:|
| full rows | **9.9348** | 9.8884 | n/a |
| `CALROWS`=128 (`M`/4) | 10.0746 (+1.41%) | **9.8804** | n/a |
| `CALROWS`=32 (`M`/16) | 11.8374 (+19.15%) | n/a | 10.0345 (+1.00%) |

Over these 8 windows, neither the paired setting nor the raised margin alone is resolvable from
the control on the converged windows. The next subsection reads both at depth. The repair
on the calibrating windows is resolved. With the margin raised, windows 1 and 2 fall from
+0.126 and +0.065 nats to +0.030 and +0.013 at `M`/4. At `M`/16 they fall from +0.519 and
+0.900 to +0.060 and +0.058.

> [!CAUTION]
> A row cut at the default `BOOTMARGIN`=1.5 clips pass two on 8.04% of columns at `M`/4 and
> 19.82% at `M`/16. Each calibrating forward returns that clipped output. The frontend's
> saturation report stays silent, because the calibrating branch passes `NULL` for
> `sat_elems`. With any `CALROWS` row cut, software must also raise `BOOTMARGIN`, to 2.25 at
> `M`/4.

The optimization is therefore `CALROWS`=`M`/4 together with `BOOTMARGIN`=2.25, two settings
changed together. Its whole cost lies in the calibration forwards, so a converged prefill does
not incur it. Neither setting is the frontend's default. `ggml_backend_rocket_mul_mat_rk3576()`
defaults to `CALROWS`=0, which runs pass one over every row, and to `BOOTMARGIN`=1.5.

> [!NOTE]
> The two calibrating windows hold the whole effect of a calibration setting. At `CALROWS`=32
> the 8-window aggregate perplexity rises 19.15%, while the paired test is 1.43 se, because 2
> windows of 8 hold all of the effect. Score the calibrating windows separately from the
> converged ones.

#### The row cut and the margin at depth

At depth, neither the row cut nor the margin changes a converged window measurably. The paired
setting `CALROWS`=128 with `BOOTMARGIN`=2.25 costs none of three models anything there. Both
settings act only on the two calibrating windows [HW sweep 2026-08-21].

**Conditions:** H96 MAX M9, `-ub` 512, so `CALROWS`=128 is exactly `M`/4, governor
`ondemand`, 2026-08-21. A 2×2 over the two settings ran on Llama-3.2-1B, at 62 converged
windows. The paired setting also ran on SmolLM2-1.7B and Qwen2.5-1.5B, at 27.

**Not characterized:** either setting alone on SmolLM2-1.7B or Qwen2.5-1.5B, because the 2×2
ran on Llama-3.2-1B only. Their readings are about the paired setting and cannot be attributed
to one setting. The `BOOTMARGIN`×`CALROWS` interior, another `-ub`, and decode are not
characterized either.

| Model | Windows | Contrast | Distance from zero | Sign count | Sign test p |
|---|---|---|---|---|---|
| Llama-3.2-1B | 3-64 | +0.0009 +/- 0.0020 se | 0.45 se | 28 of 62 worse | 0.53 |
| SmolLM2-1.7B | 3-29 | -0.0092 +/- 0.0035 se, 0.91% better | 2.59 se | 21 of 27 better | 0.0059 |
| Qwen2.5-1.5B | 3-29 | -0.0038 +/- 0.0034 se | 1.11 se | 9 of 27 worse | 0.1221 |

On Qwen2.5-1.5B the headline perplexity moves 8.5745 -> 8.5569, -0.205%.

A level is comparable across models only at a matched depth. At d=29 the three measure +0.0026
+/- 0.0030 (Llama), -0.0092 +/- 0.0035 (SmolLM2) and -0.0038 +/- 0.0034 (Qwen). They are
heterogeneous (Cochran's Q = 6.665 on 2 df), so quote them separately. No pooled figure applies, unlike
for `NCAL`=1. For an unmeasured model, use the random-effects mean, -0.0028 with tau = 0.0050.

The cut alone, at the default margin 1.5, is null on the converged windows too: -0.0011 +/-
0.0022, 23+/39-, sign test p = 0.0559. The 8.04% pass-two saturation therefore does no
measurable damage through the frozen scale. The +1.41% in the 8-window table is an aggregate
over 8 windows rather than a converged cost.

The split by phase accounts for the whole effect. The cut alone puts +0.0908 and +0.1297 nats
on the two calibrating windows, against -0.0011 on the 62 converged ones. The raised margin
repairs window 1 by 11x (+0.0908 -> +0.0083) and window 2 by 2.5x. The margin alone is +0.0002
+/- 0.0024 on the converged windows and +0.0226 on window 2. Both settings act on the
calibrating windows, and `BOOTMARGIN` repairs there what the cut does.

> [!NOTE]
> On this path an 8-window reading of a converged contrast can have the wrong sign. Five
> contrasts move 0.005-0.011 nats between d=8 and their deepest reading, and two flip sign.

The calibrating phase's damage splits differently per model, and only its sum is stable.
Qwen2.5-1.5B inverts the ordering of the two windows:

| Model | Window 1 | Window 2 | Sum |
|---|---|---|---|
| Llama-3.2-1B | +0.0083 | +0.0528 | 0.0611 |
| SmolLM2-1.7B | +0.0241 | +0.0521 | 0.0762 |
| Qwen2.5-1.5B | +0.0296 | +0.0126 | 0.0422 |

Either window alone ranges 3.6-4.2x across the three, while their sum ranges 1.8x. Test the
calibrating phase on the sum. A threshold placed on window 2 from the first two models' 1.3%
agreement fails on the third, where the setting has plainly taken effect.

#### `CALROWS` across ubatch sizes

The damage of a row cut follows the ratio of pass one's rows to `M`. The setting is an
absolute row count, so one value is a different cut at every ubatch. `ggml-rocket.cpp`
clamps the setting and never scales it (`if (rk76_calrows() > 0 && rk76_calrows() < M) M1 = rk76_calrows();`).
`CALROWS`=128 is therefore `M`/4 at `-ub` 512 and `M`/16 at `-ub` 2048.

**Conditions:** Qwen2.5-1.5B, five `llama-perplexity --chunks 8` test points at
`-c/-b/-ub` 2048.

Each test point is scored on [`Dcal`](../GLOSSARY.md), the nats added over the two `NCAL`=2
calibrating windows. The score does not depend on depth, and those windows hold all of this
setting's effect:

| `CALROWS` | ratio at `M`=2048 | `Dcal` | 8-window | converged 3-8 |
|---|---|---|---|---|
| 512 | `M`/4 | +0.1276 | +1.27% | -0.0045 +/- 0.0046 |
| 128 | `M`/16 | +0.6761 | +8.73% | -0.0010 +/- 0.0036 |
| 32 | `M`/64 | +1.9985 | +28.56% | +0.0019 +/- 0.0013 |
| 128 + `BOOTMARGIN` 2.25 | `M`/16 | +0.1052 | +0.98% | -0.0045 +/- 0.0040 |

The ratio governs, and the absolute row count does not. The `M`=512 anchors are `CALROWS`=128
-> 0.1906 and `CALROWS`=32 -> 1.4186. Against them, matched-ratio pairs agree to 1.49x at `M`/4
and 2.10x at `M`/16, while the matched-absolute pair at `M1`=128 disagrees by 3.55x.

The ratio is not exact, and the residual has one sign. At a matched ratio a larger `M` is more
forgiving, and the correction grows with the depth of the cut. Pass one's `est1` is a maximum
over the first `M1` rows, so the undershoot is `f(M1)/f(M)`, one non-decreasing curve divided
by itself. The quotient is a pure function of `M1/M` only where `f` is a power law. The
residual is the signature of a logarithmically-shaped running maximum rather than a power-law
one. A fraction of `M` is therefore the setting's right form, and its accuracy at each ubatch
still has to be measured at that ubatch.

At `-ub` 2048, `CALROWS`=128 with `BOOTMARGIN`=2.25 costs +0.98% over 8 windows and measures
-0.0045 +/- 0.0040 on the converged ones. The margin repairs 6.4x of the cut's own +8.73%, the
same mechanism measured at 11x on window 1 at `-ub` 512. The same pair measures -0.55% at
`-ub` 512. The pair is not free at the larger ubatch, although it is better than free at the
smaller. The whole difference is in the calibrating windows.

#### `CALROWS` as a fraction of `M`

Set as `M`/4, the row cut does 0.25x the damage of the absolute `CALROWS`=128 and keeps 77.7%
of its saving. At `-ub` 2048, `CALROWS`=512 is `M`/4. With `BOOTMARGIN`=2.25 beside it, the
setting costs `Dcal` +0.0266, +0.301% over 8 windows, and -0.00043 +/- 0.00437 over the
converged windows 3-8. The absolute `CALROWS`=128 at the same ubatch costs `Dcal` +0.1052.
Against the bare `M`/4 cut (+0.1276), the margin repairs 4.80x.

**Conditions:** Qwen2.5-1.5B, `llama-perplexity --chunks 8` at `-ub` 2048. The savings are the
recorded `llama-bench -ub 2048` `calibrate` figures.

The recorded savings are 28.2% of `calibrate` for `CALROWS`=512 against 36.3% for 128, which
is the 77.7% kept. Set the row cut as `M`/4 and not as a fixed row count. At `-ub` 2048 that
setting is 512.

The margin's repair factor varies with the depth of the cut: 6.43x at `M`/16 against 4.80x at
`M`/4.

#### The clip fraction as a damage predictor

The clip fraction predicts the damage along the cut axis and not across the margin axis. The
margin reaches the damage twice in the frontend. First, it sets pass two's scale
`sc2 = 127/(est1*BOOTMARGIN)`, so a larger margin saturates fewer columns. Second, it sets the
recorded ceiling `est1*BOOTMARGIN` on a column that still saturates. Only the first is visible
in the clip fraction, and the clip fraction is not a sufficient statistic for the accuracy
damage.

**Conditions:** Qwen2.5-1.5B at `M`=2048. The clip grid runs without `AROW` except where the
regime column says so, and every accuracy test point runs with it.

Raising the margin 1.5 -> 2.25 at `M`=2048:

| cut | regime | clip @1.5 | clip @2.25 | fall | damage removed |
|---|---|---|---|---|---|
| `M`/4 | no `AROW` | 8.0367% | 0.2886% | 27.9x | 79% |
| `M`/16 | no `AROW` | 19.8197% | 1.7809% | 11.1x | 84% |
| `M`/4 | `AROW`=1 | 4.3461% | **0.0836%** | **52.0x** | 79% |

The margin removes far more clipping than damage at every point. Something the saturation flag
cannot see therefore accounts for the residual damage at a raised margin.

> [!NOTE]
> A clip fraction from a 3-minute `llama-bench` [`CALMAP`](../GLOSSARY.md) readout does not
> measure the accuracy of this setting. Measure perplexity, and change `CALROWS` and
> `BOOTMARGIN` together.

`AROW` itself nearly halves the clip fraction (8.0367 -> 4.3461% at margin 1.5, 1.85x), so a
clip fraction is a statement about its activation regime.

#### Resolution of the returned output

The resolution of the returned output is a third channel of the damage, and it accounts for
only a fifth of the residual. A calibrating call returns pass two dequantized at
`sc2 = 127/(est1*BOOTMARGIN)`, so the returned int8 quantum is `est1*BOOTMARGIN/127`. Here
`est1` is pass one's per-column maximum over `M1` rows at scale `cbound`. The frontend sets
`cbound` once per weight from `sumabs` alone, and `cbound` never sees the margin.

**Conditions:** Qwen2.5-1.5B at `M`=2048. The clip fractions in the table below are `AROW`=0
readouts, and the `Dcal` values are `AROW`=1 runs.

The quantum therefore factorizes, and the two settings move it in opposite directions. The
margin coarsens it on every column, while a row cut shrinks `est1` and moves it back. No clip
fraction can see this.

The resolution index `Q/Q_ctl = (margin/1.5) * est1(cut)/est1(full)` can therefore be read from
`CALMAP` bytes with no run. The uncontaminated set is blk.0's `attn_q`/`attn_k`/`attn_v`, whose
input is the embedding through an RMSNorm with no matmul upstream. On that set at `M`=2048,
`est1(512)/est1(full)` is 0.89714 and `est1(128)/est1(full)` is 0.77888:

| test point | cut | margin | `Q/Q_ctl` | clip | `Dcal` |
|---|---|---|---|---|---|
| `q_ctl` | none | 1.5 | 1.00000 | 0.2600% | 0 by definition |
| `q_r512` | `M`/4 | 1.5 | 0.89714 | 8.0367% | +0.1276 |
| `q_shipfrac` | `M`/4 | 2.25 | 1.34570 | 0.2886% | +0.0266 |
| `q_ship` | `M`/16 | 2.25 | 1.16832 | 1.7809% | +0.1052 |
| `q_g2p25` | none | 2.25 | **1.50000** | **0.0000%** | **+0.0081** |

The in-regime clip fractions in the next subsection differ from these, because these are
`AROW`=0 readouts.

The margin-only test point clips 0 of 1500160 columns, so it isolates resolution exactly: a
1.5x coarsening of the returned output costs +0.0081 nats. Interpolated to `q_shipfrac`'s
1.34570, that is +0.0053-0.0056 under either a linear or a quadratic-noise form. It is 20-21%
of `q_shipfrac`'s +0.0266, with its clipping at the control's own level. The remaining +0.021
nats, 79-80% of the residual, is therefore the clip channel and not the returned quantum. It
lies on a curve that no linear statistic of the clipping reproduces.

Damage is not monotone in resolution. The test point `q_ship` has a strictly finer output than
`q_shipfrac` (1.16832 against 1.34570) and 4x the damage. The test point `q_g2p25` is the
coarsest buildable at margin 2.25, and it is 3.3x less damaging than `q_shipfrac`. An ordering
argument over the quantum alone predicts both backwards.

#### Separability of the calibrating-window damage

The calibrating-window damage is a separable product of the cut and the margin.

**Conditions:** Qwen2.5-1.5B at `M`=2048, `AROW`=1.

Read in the `AROW`=1 regime every `Dcal` was measured in, and with the resolution channel,
[`D_res`](../GLOSSARY.md), subtracted as `D_res = 0.0162*(Q/Q_ctl - 1)`, the test points' clip
damage is:

| test point | cut | margin | clip | mass `E` | `r` | `D_res` | `D_clip` |
|---|---|---|---|---|---|---|---|
| `q_ctl` | none | 1.5 | 0.0303% | 1.40e-05 | 1.00000 | 0 | 0 by definition |
| `q_g2p25` | none | 2.25 | 0.0000% | 0 | 1.00000 | +0.0081 | 0 |
| `q_r1024` | `M`/2 | 1.5 | 1.3775% | 1.27e-03 | 0.94636 | -0.0009 | **-0.0062** |
| `q_r1024g2p25` | `M`/2 | 2.25 | 0.0177% | 2.58e-05 | 0.94636 | +0.0068 | **-0.0174** |
| `q_shipfrac` | `M`/4 | 2.25 | 0.0836% | 1.24e-04 | 0.87151 | +0.0050 | +0.0216 |
| `q_g1p875r512` | `M`/4 | 1.875 | 0.4963% | 5.99e-04 | 0.87151 | +0.0014 | +0.0473 |
| `q_ship` | `M`/16 | 2.25 | 0.9604% | 1.39e-03 | 0.74417 | +0.0019 | +0.1033 |
| `q_r512` | `M`/4 | 1.5 | 4.3461% | 4.53e-03 | 0.87151 | -0.0021 | +0.1297 |
| `q_r128` | `M`/16 | 1.5 | 17.1747% | 2.41e-02 | 0.74417 | -0.0041 | +0.6803 |
| `q_r32` | `M`/64 | 1.5 | 38.3480% | 8.44e-02 | 0.62576 | -0.0061 | +2.0046 |

The product `D_clip = g(cut)*h(margin)` fits the six cut test points at `M`/4 and deeper to a
worst residual of 2.31% (rms 1.86%). The cut factor `g` is 0.1327 at `M`/4, 0.6649 at `M`/16
and 2.0046 at `M`/64. The margin factor `h` is 1 at margin 1.5, 0.3561 at 1.875 and 0.1590 at
2.25.

The cut factor `g` is not a power law in the cut: it rises 5.01x, then 3.01x, per 4x
reduction.

#### The margin factor `h`

A power law in the margin for `h`, fitted inside [1.5, 2.25], extrapolates to margins 1.25 and
1.00 within 7.3%. Its exponent tracks the K set. Below 1.5 the `h` values contain a modeled
`D_res` whose form is wrong there (§"`BOOTMARGIN` and the no-cut response"). Read directly,
`Dcal` is not one power law over the margin (§"The direct (cut, margin) surface").

**Conditions:** Qwen2.5-1.5B, `-ub` 2048, `--chunks 8`.

The frontend resets a `BOOTMARGIN` below 1.0 to 1.5 (`bootmargin < 1.0f`), so 1.0 is the
lowest margin it runs. Two test points at cut `M`/4 and margins 1.25 and 1.00 measure the
extrapolation to that minimum [HW sweep]:

| cut | margin | clip | `Dcal` | PPL[8] vs control | implied `h` | power law |
|---|---|---|---|---|---|---|
| `M`/4 | 1.00 | 77.171% | **+0.92760** | +12.347% | **7.3094** | 6.9407 |
| `M`/4 | 1.25 | 25.672% | **+0.32464** | +4.056% | **2.5645** | 2.3897 |

The power law `h(m) = (m/1.5)^-4.778`, fitted entirely inside [1.5, 2.25], predicts a 1.5x
extrapolation to the minimum within 5.3% and 7.3%. Both measured values sit above it, so the
true `h` is slightly steeper still. The ratio `Dcal`(1.0)/`Dcal`(1.25), which is free of `g`
and of `D_res`, is 2.857 against the power law's 2.912.

The exponent differs between architectures. SmolLM2-1.7B at the same cut and margin 1.0
measures `Dcal` = +0.97094 (PPL[8] 8.3207 against its own control 7.4265, +12.041%). Over the
same [1.0, 2.25] range its margin ratio (the [margin index](../GLOSSARY.md) `R`, defined in the
next subsection) is 30.60 against Qwen's 50.57, 1.65x smaller. Its exponent is 4.22 against
4.84, 1.15x shallower [host-computed].

> [!NOTE]
> A cross-model comparison of margin cost taken at one end of the margin axis does not hold at
> the other. At margin 2.25 SmolLM2-1.7B's `M`/4 damage is 1.99x Qwen2.5-1.5B's, and at margin
> 1.00 the two models are 4.7% apart.

The exponent is shared between two models that share a K set. In the table in the next
subsection, 4.230 and 4.219 agree to 0.26% across a tokenizer, a corpus, 8 blocks of depth and
GQA-against-MHA. A same-family model at a different K set measures 3.223. The exponent is
therefore a property of the K set, or of what co-varies with it, and not of the architecture
or the model.

The clip fraction at a new margin can be computed off-line from a `CALMAP`'s own `est2/est1`
distribution. It gave 78.2% and 26.0% for these two test points, which measured 77.171% and
25.672%, within 1.3%.

#### The margin index across architectures and K sets

Two models that share a K set have the same margin index `R`, and a model at another K set
does not [HW sweep 2026-08-22]. `R` is `D_clip(cut M/4, 1.00) / D_clip(cut M/4, 2.25)`,
where `D_clip(m)` is `Dcal(cut, m)` minus the no-cut `D_res(m)`. `D_res` is the `Dcal` of the
no-cut test point at margin 2.25.

**Conditions:** `-ub` 2048, `--chunks 8`, `rocket` 1.6.0, 2026-08-22.

Llama-3.2-1B-f16 has the llama architecture and SmolLM2's exact `n_embd` 2048 / `n_ff` 8192, at
16 blocks against 24. It runs its own control plus three test points:

| model | arch | blocks | `n_embd`/`n_ff` | `head_count_kv` | ctl PPL[8] | `D_res` | R over [1.00, 2.25] | exponent |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | qwen2 | 28 | 1536/8960 | 2 | 7.2534 | +0.00811 | 50.57 | 4.838 |
| SmolLM2-1.7B | llama | 24 | 2048/8192 | 32 | 7.4265 | +0.02060 | 30.60 | 4.219 |
| Llama-3.2-1B | llama | 16 | 2048/8192 | 8 | 11.4024 | **+0.01626** | **30.87** | **4.230** |
| SmolLM2-360M | llama | 32 | **960/2560** | 5 | 9.5608 | **+0.01484** | **13.65** | **3.223** |

The margin index is identical on the two llama models that share a K set. It measures 30.87
against 30.60, with exponents 4.230 against 4.219. These agree to 0.9% and 0.26% across an
8-block depth difference and across GQA-against-MHA, while Qwen sits 1.64x away. R is a ratio
at one cut, so both models' `g` and both models' mixed-`M` populations cancel out of it.

SmolLM2-360M differs from SmolLM2-1.7B in the K set, and shares with it:

- The architecture
- The `tokenizer.ggml.pre` smollm over an identical 49152 vocabulary, so the same corpus
  segments into the same windows (true of no other pair here)
- The training family
- The base-checkpoint status

Its R is 13.65, 0.446x the anchor's. The two pairs separate the K set from what else moves, with
no fit:

| pair | what moves | R ratio |
|---|---|---|
| SmolLM2-1.7B vs Llama-3.2-1B | tokenizer, vocabulary, corpus, depth 24 vs 16, `head_kv` 32 vs 8, base vs instruct | **1.009x** |
| SmolLM2-1.7B vs SmolLM2-360M | the K set, 2048/8192 vs 960/2560 (depth 24 vs 32, `head_kv` 32 vs 5) | **0.446x** |

R follows the K set. Both `n_ff` and `n_ff`/`n_embd` are monotone over these four points, and
a fifth model separates them and refutes both, below.

The pair does not measure the K set alone. SmolLM2-360M K-splits none of its GEMMs, because 960
and 2560 are both `rk76_k_single_ok`. All three anchors split exactly one of the seven per block
(`ffn_down`, K >= 6176). Model scale co-varies as well. The move is therefore "the K set or the
split population or scale", jointly.

The flag `ROCKET_RK3576_FORCESPLIT=1` separates the split. A fifth model separates the other
two: SmolLM2-135M, at the same `n_ff`/`n_embd` ratio and 0.600x the scale. It splits nothing
either, so the split population is held fixed there. It refutes the ratio and `n_ff` alone, below.
Model scale orders the damage one way on the default walks and the other way at a fixed chunk
depth (§"Three checkpoints at a fixed chunk depth").

The resolution channel is the model-specific one, and it moves where the margin index does
not. Llama-3.2-1B's `D_res` is 0.789x SmolLM2's and 2.004x Qwen's. It moves with depth, but
nowhere near proportionally: 0.789x for a 0.667x block ratio.

SmolLM2-360M dissociates the two channels on the K axis. Its `D_res` is 0.720x its
same-family anchor's, while its R is 0.446x. The resolution channel therefore does not follow
the K set either. It stays per-model over four points spanning 2.54x (+0.00811 to +0.02060).

One per-model scalar setting both channels is not supported: between SmolLM2-1.7B and
Llama-3.2-1B the resolution channel moves 0.789x while the margin index moves 1.009x.

> [!NOTE]
> Two properties of Llama-3.2-1B affect any figure transported to it. Its `head_count_kv` is 8
> against SmolLM2's 32, so two of the seven projections a block offloads have `N`=512 rather
> than 2048. The pair is therefore not a pure depth contrast. Also, 24.35% of its calibrated
> columns sit at `M`=1024 (309760 of 1272320), against SmolLM2's 8.57% and Qwen's 19.0%. On a
> quarter of them `CALROWS`=512 is an `M`/2 cut, and the "cut `M`/4" label covers only 75.65%.
>
> Both properties cancel out of R, and neither cancels out of a level. Llama-3.2-1B's
> Dcal(cut 512, 1.00) = +0.90033 is therefore not comparable with Qwen's +0.92760 or SmolLM2's
> +0.97094.

The clip fraction at cut 512 / margin 1.00 is nearly model-independent: 75.040% on
Llama-3.2-1B, against Qwen's 77.171% and SmolLM2's 76.14%. Llama-3.2-1B's margin-2.25 no-cut
test point clips 0 of 1272320 columns, as on both other models. Its `D_res` is therefore a
measurement rather than a response minus a model.

#### The denominator of the margin index

The margin index is the reciprocal of a small difference, and a fifth model puts that
difference at zero [HW sweep 2026-08-22]. The denominator of `R`, `D_clip(cut M/4, 2.25)`, is a
difference of two same-signed quantities of order 0.02 nats, and nothing constrains it to stay
positive.

**Conditions:** `-ub` 2048, `--chunks 8`, `rocket` 1.6.0, 2026-08-22.

SmolLM2-135M is a llama at 576/1536 with 30 blocks and `head_kv` 3. It uses smollm over the
same 49152 vocabulary and is the HuggingFaceTB base checkpoint. Its `n_ff`/`n_embd` is
2.666667, exactly SmolLM2-360M's, at 0.600x its width. It puts the denominator at -0.00338:

| model | ratio | `D_res`(2.25) | `Dcal`(cut, 1.00) | `Dcal`(cut, 2.25) | `D_clip`(2.25) | `R` | raw ratio |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 5.8333 | +0.00811 | +0.92760 | +0.02656 | +0.01845 | 50.57 | 34.92 |
| SmolLM2-1.7B | 4.0000 | +0.02060 | +0.97094 | +0.05278 | +0.03218 | 30.60 | 18.40 |
| Llama-3.2-1B | 4.0000 | +0.01626 | +0.90033 | +0.04577 | +0.02951 | 30.87 | 19.67 |
| SmolLM2-360M | 2.6667 | +0.01484 | +0.69288 | +0.06634 | +0.05150 | 13.65 | 10.44 |
| SmolLM2-135M | **2.6667** | **+0.02332** | **+0.51870** | **+0.01994** | **-0.00338** | **-158.11** | **26.02** |

The four anchors' 3.7055x spread in `R` decomposes exactly into a 1.3276x numerator move and a
2.7911x denominator move. The reciprocal of that small difference is therefore 78% of the log
spread. The four values of `R` (50.57, 30.87, 30.60 and 13.65) differ mostly through the denominator.

> [!NOTE]
> Read the denominator column, not the `R` column. Fit `Dcal` directly on (cut, margin). Do
> not derive `D_clip`, and do not quote `R` for a model whose denominator has not been read.

On the fifth model, the clip term at margin 2.25 is consistent with zero, not negative. Its two
calibrating windows disagree in sign, -0.00709 and +0.00371, where SmolLM2-360M's agree
(+0.00155, +0.04995). The net is 38x the reconstruction precision, so the arithmetic is exact.
The quantity itself is unresolved, and an index dividing by it is undefined rather than large.

The mechanism is the two channels netting out. Margin 2.25 coarsens the returned quantum on
every column, while a row cut shrinks `est1` and moves it back the other way. On a model whose resolution cost is
high and whose clipping is cheap, the cut therefore partly offsets the margin.

The margin channel itself is intact on this model. At the same cut, margin 1.00 costs +0.51870
against margin 2.25's +0.01994, a 26.0x ratio in the same direction as all four anchors. A
negative `R` here means the index failed, not that the channel reversed.

The ratio `n_ff`/`n_embd` does not set the index, on the one statistic that survives a zero
denominator. The raw `Dcal(cut, 1.00)/Dcal(cut, 2.25)` is 26.02 against SmolLM2-360M's 10.44 at
an identical ratio, 2.49x apart. The raw statistic is also non-monotone in `n_ff`: 34.9 at
8960, 18.4 at 8192, 19.7 at 8192, 10.4 at 2560 and 26.0 at 1536. The non-monotonicity does not
extend to model scale. The raw ratio is itself 76% denominator, and on its numerator alone
scale is monotone (next subsection).

The resolution channel does not scale with width either. On the smallest model `D_res` is
+0.02332, above all four anchors. Over five models it spans 2.875x with no monotone relation to
width, depth, `n_ff` or baseline loss. Its no-cut margin-2.25 test point clips 0 of 409344
columns, so this is a measurement and not a response minus a model.

The clip fraction holds on a fourth model: 80.859% at cut 512 / margin 1.00. The other three
measure 77.171% (Qwen), 75.040% (Llama) and 77.351% (SmolLM2-360M). It is nearly
model-independent at a given (cut, margin), 1.078x over four models spanning 5.8x in `n_ff`,
and read in its own prompt regime.

#### The direct (cut, margin) surface

Read directly, the (cut, margin) surface shows that the `D_res` subtraction is sound where
clipping dominates, and that the raw ratio has the index's defect. All 32 calibration test
points at `-ub` 2048 are read here as one response, `Dcal(cut, margin)`, with no `D_clip` and
no index [HW sweep].

**Conditions:** five models, `-ub` 2048, `rocket` 1.6.0, one corpus, calibrating windows only.
The levels are comparable within a tokenizer and nowhere else.

**Not characterized:** another corpus or another ubatch.

The `D_res` subtraction holds where the clip channel dominates, and the division fails, not the
subtraction. Separability predicts that `D_clip(cut, 1.50)/D_clip(cut, 2.25)` does not depend
on the cut. The term `D_res(1.50)` is zero by construction, because the control is margin 1.5
with no cut.

On Qwen2.5-1.5B the unsubtracted margin ratio at cuts 128 and 512 is 6.427 and 4.805, 1.337x
apart. Subtracting the independently measured no-cut +0.00811 makes them 6.963 and 6.918,
1.0066x apart. The reduction is 51x, from one constant that was never fitted to either test
point. At cut 1024 both readings are negative and 18.6x apart. A `D_clip` is therefore a sound
quantity at `M`/4 and deeper, and an unsound one near its own zero. The defect is the
reciprocal, not the difference.

The sign of `D_clip(M/4, 2.25)` depends on which side of a minimum the cut falls. Along margin
2.25 the two models measure:

| Cut | Qwen2.5-1.5B | SmolLM2-1.7B |
|---|---|---|
| 128 | +0.10521 | n/a |
| 512 | +0.02656 | +0.05278 |
| 768 | +0.00931 | n/a |
| 1024 | -0.01065 | +0.03770 |
| 1536 | +0.00335 | +0.02405 |
| 2048 | +0.00811 | +0.02060 |

Qwen crosses its own no-cut level just above 768 and bottoms near `M`/2. SmolLM2-1.7B is
monotone over the cuts it has, so its crossing is above `M`*0.75. SmolLM2-135M's `M`/4 test
point already sits below its no-cut test point, so its crossing is below `M`/4. The crossing
point is model-dependent, and it is what flips the denominator's sign.

The raw ratio has 76% of its spread in the denominator, the same defect as `R`'s 78%.
Decomposed at the ends of the five-model sequence (Qwen over SmolLM2-360M), the numerator moves
1.3388x and the denominator 2.4973x. Their product, 3.3433x, is the raw ratio exactly.

On the numerator alone, model scale is monotone on the default walks, and it is the one
architectural candidate those walks leave. The test point `Dcal(cut M/4, margin 1.00)` is the
largest in the corpus (0.52-0.97 nats against a reconstruction precision of 8.79e-5) and
subtracts no model. It is a level, so only
the three checkpoints that share the smollm tokenizer segment the corpus identically and are
comparable:

| model | params | `n_embd` | `n_ff` | blocks | `n_ff`/`n_embd` | calibrated columns | `Dcal`(`M`/4, 1.00) |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M | 0.135B | 576 | 1536 | 30 | 2.6667 | 409344 | +0.51870 |
| SmolLM2-360M | 0.360B | 960 | 2560 | 32 | 2.6667 | 651264 | +0.69288 |
| SmolLM2-1.7B | 1.710B | 2048 | 8192 | 24 | 4.0000 | 1769472 | +0.97094 |

The numerator is monotone in parameters, `n_embd`, `n_ff` and the calibrated-column count. It
is non-monotone in depth (30, 32 and 24 blocks). The two checkpoints at an identical
`n_ff`/`n_embd` differ 1.336x, so the ratio does not set the numerator either. The raw
statistic that argues against model scale is three-quarters denominator, so it does not exclude
scale. At a fixed chunk depth the same three checkpoints order the other way (§"Three checkpoints
at a fixed chunk depth"). Scale therefore does not set the numerator independently of the chunk
walk.

The clip fraction is not the driver. At that same test point it measures 80.859, 77.351,
77.171 and 75.040% on four models, 1.078x, while the response spans 1.872x. The
calibrated-column count is what remains collinear with scale on every model here. Separating
the two needs a test point that moves the population at a fixed checkpoint (next subsection).

The margin axis is not a power law on the direct response. Qwen at cut `M`/4 over five margins
measures:

| Margin | `Dcal` | Local log-log exponent to the next margin |
|---|---|---|
| 1.00 | +0.92760 | -4.705 |
| 1.25 | +0.32464 | -5.120 |
| 1.50 | +0.12764 | -4.318 |
| 1.875 | +0.04870 | -3.325 |
| 2.25 | +0.02656 | n/a |

The local exponent is non-monotone, with a 1.54x spread, and one power law over the range
misses by 1.230x at its worst point. The direction and the end-to-end size hold: 34.9x between
the two ends at that cut on that model.

The two channels dissociate with no index at all. SmolLM2-135M has the lowest numerator of the
five (+0.51870, 0.534x the highest) and the highest `D_res` (+0.02332, 2.875x the lowest). That
one model shows, with no fit and no derived quantity, that one per-model scalar does not set
both channels.

#### The calibrated-column population

The calibrated-column population is not the driver: 4.09x more columns give more clipping and a
quarter of the damage [HW sweep 2026-08-22]. The setting
`ROCKET_RK3576_FORCESPLIT=1 ROCKET_RK3576_KCHUNK=512` cuts every offloaded K into 512-chunks
(960 -> 512+256+128+64, 2560 -> 5×512). A calibrating call logs one column per output column
per chunk. SmolLM2-360M's calibrated population therefore moves 651264 -> 2666496, 4.0944x, at
a byte-identical checkpoint, tokenizer, corpus and window segmentation. The split breaks the
collinearity between model scale and column count that every other model here has.

**Conditions:** SmolLM2-360M, `-ub` 2048, `--chunks` 4, governor `ondemand`, 2026-08-22. The
split moves the population and the contraction depth together, so it bounds what a population
term can contribute and does not subtract it.

| test point | PPL[4] | `Dcal` (windows 1-2) | clipped |
|---|---|---|---|
| `q_fsctl` (the split's own control) | 11.3676 | n/a | 0 of 2666496 |
| `q_fsr512g1p0` (cut 512, margin 1.00) | 11.8050 | **+0.16382** | 2250316 (**84.392%**) |
| `q_fsg2p25` (no cut, margin 2.25) | 11.3775 | **+0.00448** | 0 |

Against the same checkpoint's unsplit +0.69288, the cut-512 margin-1.00 damage under 4.09x the
columns is 0.2364x, down. The clip fraction goes up 1.091x (84.392% against 77.351%). More
columns, with more of them clipping, give a quarter of the damage. A term that grows with the
calibrated-column count cannot produce that.

Both channels fall at this chunk size. The resolution term `D_res` is +0.00448, 0.3016x the
unsplit +0.01484, against the clip channel's 0.2364x. A second chunk size separates them again
(next subsection).

The split's own arithmetic does not account for the change. The FORCESPLIT control measures
PPL[4] 11.3676 against the unsplit control's 11.4391, 0.9937x, so the chunked baseline is, if
anything, slightly better on this model.

The K-split's recorded 1.60x perplexity cost belongs to a different configuration. It is the
cost of turning `KSPLIT` on for a `K >= 6176` `ffn_down` that is otherwise declined to the CPU.
Nothing on this model reaches that bound.

#### Clip damage at a second chunk size

The clip damage follows the chunk depth, not the chunk count, and at a second chunk size the two
channels come apart again. `ROCKET_RK3576_KCHUNK=256` on the same checkpoint takes the
population to 3563520 columns. That population is 5.4717x the default path's 651264 and 1.3364x
the 512-chunk test point's.

**Conditions:** SmolLM2-360M, `ROCKET_RK3576_FORCESPLIT=1`, three test points at `--chunks` 4
and `-ub` 2048.

**Not characterized:** a third chunk size, another model.

| | unsplit | `KCHUNK`=512 | `KCHUNK`=256 |
|---|---|---|---|
| calibrated columns | 651264 | 2666496 | 3563520 |
| column-weighted chunks a GEMM | 1 | 4.0943 | 5.4717 |
| deepest chunk | 2560 | 512 | 256 |
| `Dcal`(cut 512, margin 1.00) | +0.69288 | +0.16382 | **+0.07245** |
| `Dcal` as a fraction of unsplit | 1.0000 | 0.2364 | **0.1046** |
| clip fraction of that test point | 77.351% | 84.392% | **85.271%** |
| damage per clipped column, nats | 1.375e-06 | 7.280e-08 | **2.384e-08** |
| `D_res` = `Dcal`(no cut, 2.25) | +0.01484 | +0.00448 | **+0.01013** |

The damage does not follow the reciprocal of the chunk count. Between the two split test points
the local exponent on the count is 2.8135. A reciprocal law gives an exponent of 1.

The maximum chunk depth is the variable that fits. Halving `KCHUNK` halves the deepest chunk
exactly, while it raises the column-weighted count only 1.3364x. The greedy walk is *ragged*,
so count and maximum depth are not reciprocal, even though count and mean depth are reciprocal
by construction. The measured fall is 0.4423x, against 0.5000x for a halved depth (1.13x) and
0.7483x for the count (1.69x). The segment exponents are 1.177 and 2.814. The residual is in
the direction of super-linearity in depth [two split points, one model].

`D_res` reverses, so the contraction depth does not name a quantity the two channels share. At
`KCHUNK`=256 it is +0.01013, which is 2.2614x its `KCHUNK`=512 value and 0.6827x the unsplit
value. `D_res` is therefore *non-monotone* in the chunk count, with a minimum at `KCHUNK`=512,
while the clip channel falls 0.4423x over the same step. Two competing terms are visible, and
neither combination reproduces the size. Against a measured 2.2614x, `count*quantum` gives
1.277x, `count*quantum²` gives 1.220x and `count²` gives 1.786x [hypothesis, named, not
isolated].

The cross-tab by chunk depth is monotone in both columns. Over the control test points' own
CALMAPs it measures:

| `Kc` | mean `est1` | clip fraction of the cut test point |
|---|---|---|
| 64 | 69528 | 87.742% |
| 128 | 85110 | 85.586% |
| 256 | 94860 | 82.961% |
| 512 | 128443 | 82.346% |

A shallower chunk is quantized finer. The returned quantum is `est1*margin/127`, and `est1`
falls 0.5413x from `Kc`=512 to `Kc`=64 inside one run. A shallower chunk also clips more,
because a partial over fewer terms has a heavier relative tail. The second cause is that a
512-row maximum undershoots the 2048-row maximum more often.

> [!NOTE]
> `Kc` alone does not identify a column population. The `Kc`=256 columns measure `est1` 94860
> at `KCHUNK`=512 and 102305 at `KCHUNK`=256, because they sit at different offsets into K in
> the two walks.

The tail of the K=960 walk is a control the split does not reach. The walk ends 128+64 at both
chunk sizes, so those 589824+589824 columns are the identical population in both test points.
They measure identically (`est1` 1.0002x and 0.9997x, clip 1.0008x and 1.0004x). The two test
points therefore differ *only* in their deep chunks. A common additive term can only pull a
ratio toward 1, so the measured 0.4423x is an upper bound on the deep chunks' own fall.

#### Three checkpoints at a fixed chunk depth

At a fixed maximum chunk depth of 512, the per-column clipping converges across three SmolLM2
checkpoints and the calibrating-window damage does not [HW sweep 2026-08-22]. The damage levels
span 2.637x, wider than the 1.8719x they span on the default walks, and their order inverts. At
a fixed depth the damage falls monotonically with model scale, and on the default walks it
rises.

**Conditions:** RK3576, SmolLM2-135M, SmolLM2-360M and SmolLM2-1.7B,
`ROCKET_RK3576_FORCESPLIT=1 ROCKET_RK3576_KCHUNK=512`, `-ub` 2048, `--chunks` 4, governor
`ondemand`. The three share the smollm tokenizer, which is the only reason their levels are
comparable.

The two settings give each checkpoint a maximum chunk depth of exactly 512. The walks are
`576 -> 512+64` and `1536 -> 3x512` on the 135M, `960 -> 512+256+128+64` and `2560 -> 5x512` on
the 360M, and `2048 -> 4x512` and `8192 -> 16x512` on the 1.7B. The default walks' maxima are
1536, 2560 and 2304. Only the maximum is held equal, and the depth distribution differs. On the
135M, 43.9% of the columns sit at `Kc`=64, the 360M has 66.4% below 512, and the 1.7B is
uniform at 512.

| | SmolLM2-135M | SmolLM2-360M | SmolLM2-1.7B |
|---|---|---|---|
| `Dcal`(cut 512, margin 1.00), default walk | +0.51870 | +0.69288 | +0.97094 |
| the same at `KCHUNK`=512 | **+0.36230** | **+0.16382** | **+0.13742** |
| as a fraction of its default-walk value | 0.6985 | 0.2364 | **0.1415** |
| column-weighted chunk count `C` at `KCHUNK`=512 | 2.0844 | 4.0943 | 4.8571 |
| `p` in `FALL = C^-p` | 0.4886 | 1.0231 | **1.4710** |
| clip fraction of that run | 84.481% | 84.392% | 82.727% |
| `D_res` = `Dcal`(no cut, 2.25) at `KCHUNK`=512 | +0.00758 | +0.00448 | +0.00750 |
| as a fraction of its default-walk value | 0.3252 | 0.3019 | 0.3640 |

The maximum chunk depth therefore does not set the damage level. No single power of the chunk
count fits the three either, because `p` is itself monotone in scale over a 3.01x range. Model
scale does not set the level independently of the chunk walk. Ordered by scale, the three rise on
the default walks and fall at `KCHUNK`=512. The split's effect, the fall and `p`, is monotone in
scale, so a scale term that depends on the walk is not excluded.

The per-column statistics do converge at a fixed depth. On the `Kc`=512 sub-population, which
has the same chunk depth in all three walks, the runs with a row cut give:

| `Kc`=512 sub-population | SmolLM2-135M | SmolLM2-360M | SmolLM2-1.7B | spread |
|---|---|---|---|---|
| clip fraction | 81.985% | 82.346% | 82.727% | 1.0091x |
| mean `est1` | 117277 | 128443 | 115208 | 1.1149x |

The three checkpoints differ 12.7x in parameter count, and the chunk depth holds these per-column
statistics together across them. The damage those columns do to the loss is 2.637x apart at the
same time. The chunk depth sets what the split does to a column, and does not set what the split
does to the loss.

The resolution channel and the clip channel respond to the split differently. `D_res` falls by
a nearly common factor on all three models, 0.3252, 0.3019 and 0.3640, a 1.21x spread. The clip
channel falls 0.6985, 0.2364 and 0.1415, a 4.94x spread. One multiplicative response to the
split describes one channel and not the other.

#### The `KCHUNK` sweep on SmolLM2-135M

On SmolLM2-135M, the calibrating-window damage under a forced K-split follows the maximum chunk
depth of the `n_embd`-K GEMMs [HW sweep]. It follows that depth as an ordering with a residual,
with no stable exponent. Of the 30.2% fall in damage between the unsplit run and `KCHUNK`=512,
those GEMMs account for 91% and the `ffn_down` split for 9% [host-computed].
Neither the global maximum chunk depth nor the chunk count sets the damage.

**Conditions:** RK3576, SmolLM2-135M, `-c/-b/-ub` 2048, `--chunks` 4, `KCHUNK` swept. Each
setting is scored against its own control, because a different chunk size is a different
arithmetic.

| `KCHUNK` | `K`=576 walk | `K`=1536 walk | max depth | `C` | columns | `Dcal`(512, 1.00) | `D_res` |
|---|---|---|---|---|---|---|---|
| off | 576 | 1536 | 1536 | 1.0000 | 409344 | **+0.51870** | +0.02332 |
| 2048 | 512+64 | 1536 | **1536** | 1.9156 | 784128 | **+0.37641** | +0.02164 |
| 1024 | 512+64 | 1024+512 | 1024 | 2.0000 | 818688 | **+0.37081** | −0.00783 |
| 512 | 512+64 | 3x512 | 512 | 2.0844 | 853248 | **+0.36230** | +0.00758 |
| 256 | 256+256+64 | 6x256 | 256 | 3.2533 | 1331712 | **+0.16241** | −0.00257 |
| 128 | 4x128+64 | 12x128 | 128 | 5.5909 | 2288640 | **+0.06283** | +0.00343 |

`KCHUNK`=2048 and the unsplit control have the same maximum chunk depth, and `Dcal` falls 27.4%
between them. At 2048 the greedy walk leaves `K`=1536 whole, because 2304 and 2048 both exceed
it and the walk picks 1536, while `K`=576 still splits `512+64`. The global maximum chunk depth
therefore does not set the damage. `KCHUNK`=64 is refused: `ffn_down` needs 24 chunks, against
the walk's 16-chunk cap.

`ffn_down` is nearly inert. Its whole split takes one chunk of 1536 down to `3x512`, a 3x cut in
the program's deepest chunk over 12.2% of the columns. The cut is worth 3.7% of the
`KCHUNK`=2048 setting's `Dcal`. The `n_embd`-K family (q/k/v/o/gate/up and the output head)
accounts for 27.4 points of the 30.2% fall to `KCHUNK`=512, 91%, on 87.8% of the columns
[host-computed]. The difference between the 1024 and 512 settings lies only inside `ffn_down`'s
share.

The ratio across one step transfers between checkpoints where no level does. The 512 -> 256
step's ratio is 0.44827 on the 135M against 0.44225 on the 360M, 1.3% apart. The level at a
fixed depth, by contrast, is 2.637x apart across the three checkpoints.

The mean `est1` of `ffn_down` falls monotonically with its chunk depth `Kc`:

| `Kc` | 1536 | 1024 | 512 | 256 | 128 |
|---|---|---|---|---|---|
| `ffn_down` mean `est1` | 185839 | 148655 | 132503 | 108084 | 89437 |

`KCHUNK`=320 holds the `n_embd` family at exactly two chunks and a mean depth of exactly 288,
the same as at 512. The greedy walk takes `K`=576 to `320+256` where 512 takes it to `512+64`,
and (512+64)/2 = (320+256)/2, while the maximum depth moves 512 -> 320. `Dcal` falls 2.06x,
+0.36230 -> +0.17629. The chunk count and the mean depth therefore do not set the damage either.

The two designed pairs bound the answer from opposite sides. One shares a global maximum depth
and differs 27.4%, and the other shares a chunk count and a mean depth and differs 2.06x.

Over the seven settings, a power law in the `n_embd` maximum depth fits to a worst factor of
1.1624x. A law in the global chunk count fits to 1.3934x. The three runs that share an `n_embd`
depth of 512 measure within 3.89%, while their global `C` spans 8.81%. The local exponent still
runs 2.72, 1.53, 0.37 and 1.37 over the segments. The data therefore support an ordering in that
depth with a residual, and no exponent. The regression's 1.20x advantage is weak beside the two
designed pairs, which establish the result.

A chunk depth that is not a power of two takes a narrower rotation. At `Kc`=320, `rk_hadamard_ok`
is false. The rotation is then the block-diagonal fallback at b = `Kc & -Kc` = 64, narrower than a
full H of the chunk's own width. The rotation is orthonormal, so the exact per-chunk accumulator is
unchanged. Because `Dcal` differences two runs at the same `KCHUNK`, the width cancels to first
order [expected]. It survives as a second-order term, since `est1` is read off the quantized
product and not the exact one [expected].

`Kc` is not a population label: the per-`Kc` clip ordering inverts within a single run. At
`KCHUNK`=320, `Kc`=256 clips 81.153% against `Kc`=320's 82.457%, so the shallower chunk clips
less, the opposite of the 512 setting's 87.667 against 81.985. The mean `est1` inverts with it
(106069 against 87610, where the deeper chunk sums more terms). The two chunks cover different
k-ranges (`n_embd` k[0,320) against k[320,576)), and the range outweighs the depth. The
whole-run clip at 320 is 81.879%, below every other split setting.

`D_res` collapses only when both families are split. At `KCHUNK`=576 the `n_embd`-K family takes
a single chunk: the forced branch of `rk76_ksplit` returns the same one-element list as its
non-forced branch. Then 73.1% of the columns run the unsplit run's program, though only
block 0's are identical in value (below). The setting measures `ffn_down`'s 1 -> 4 chunk step
alone:

| | `ffn_down` whole | `ffn_down` split |
|---|---|---|
| **`n_embd` whole** | +0.02332 (unsplit) | **+0.02376** (`KCHUNK`=576) |
| **`n_embd` split** | +0.02164 (`KCHUNK`=2048) | −0.00783 … +0.01187 (four settings) |

Splitting `ffn_down` four ways with `n_embd` whole moves `D_res` by +0.00044, which is 1.9% of
the collapse the both-split settings show. Splitting `n_embd` alone moves it −0.00168.

The family shares are not additive. At `KCHUNK`=576, `Dcal` measures +0.51361, an `ffn_down`
step of −0.00509, against the −0.01411 the same tensor costs with `n_embd` split. The ratio is
0.360x, and the step is 0.98% of its own baseline against 3.75%, so a scale effect does not
explain it. A share measured for a tensor family depends on the other family's state, not on
the tensor alone.

The 0.360x is not an interaction of the two families: the matched-count `KCHUNK`=832 setting
measures the same step at +0.00689, the opposite sign. The two `ffn_down` steps differ because
they are different chunk counts on a curve that is not monotone in the count.

> [!NOTE]
> Most `KCHUNK` values reproduce another value's walk. On SmolLM2-135M, 182 values of `k` pass
> `rk76_k_single_ok` and give a terminating walk. The greedy walk collapses 137 of them onto the
> `KCHUNK`=2048 walk, and only 16 distinct `C` exist. `KCHUNK`=1536 is one of the 137, a
> bit-identical replicate rather than a new test point. Print the whole map of walks before
> choosing a value.

The bit-identity is the program's, not the columns'. Block *L*'s `n_embd` GEMMs read
activations that passed through block *L*−1's `ffn_down`. Once the other family is split, only
block 0's columns are identical in value. Five settings share the `n_embd` walk [512,64]:
`KCHUNK` 2048, 1024, 768, 832 and 512. Their `frozen` group's clip rate spans 84.174% to 85.629%,
a 1.455 pp spread, on identical arithmetic over identical 749568 columns.

A per-`Kc` clip rate obtained by inverting a whole-run clip rate is not quotable. The inversion
has a `d/dr` factor of 10.84 here, so the 1.455 pp spread becomes 15.8 pp on the result. Any gap
read off one such rate is inside its own substitution error. The within-run comparisons that
show `Kc` is not a population label are unaffected. Every
`Dcal` and `D_res` figure above is also unaffected, since each is a difference between whole
runs.

#### The `ffn_down` chunk-count dose-response

With the `n_embd` family held at [512,64], `Dcal` is not monotone in `ffn_down`'s chunk count
[HW sweep]. The column-weighted chunk count `C` does not set the control's perplexity level
either.

**Conditions:** RK3576, SmolLM2-135M, configured as in §"The `KCHUNK` sweep on SmolLM2-135M".
The settings `KCHUNK` 832 and 768 come from the one walk group that holds `n_embd` at [512,64]
and moves only `ffn_down`. Each is therefore a matched pair against a run in that section.

**Not characterized:** what sets the cross-model level, and whether the interaction belongs to
the two families or to the chunk count against the baseline. The minimum chunk depth alone is
not characterized either. The settings `KCHUNK` 960 ([960,512,64]) and 896 ([896,512,128]) join
512 ([3x512]) as a three-setting family at `C`=2.08443, count 3 and `n_embd` [512,64]. The
family differs only in the minimum chunk depth, a pair structure that alone has almost no
effect at `KCHUNK`=768.

`KCHUNK`=832 gives `ffn_down` [832,512,128,64], four chunks with `n_embd` split, the
matched-count complement of the `KCHUNK`=576 setting. It measures `Dcal` = +0.38330, a 1 -> 4
step of +0.00689 against −0.00509 for the same step with `n_embd` whole. The two steps have
opposite signs. The fixed-`n_embd` dose-response is:

| chunk count | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| `Dcal` | +0.37641 | +0.37081 | +0.36230 | **+0.38330** |

The response reverses at the fourth chunk and ends above the one-chunk run. The 832 run is the
only one on this model whose `ffn_down` walk goes below 128. It is also the only one with a
64-deep chunk, which is the leading candidate for the reversal [hypothesis].

> [!NOTE]
> Do not fit a `Dcal` law in the chunk count. The dose-response reverses at the fourth chunk,
> past any extrapolation of the first three.

`KCHUNK`=768 gives `ffn_down` [768,768], which matches the `KCHUNK`=1024 setting's `C`
(2.00000), chunk count (2), column population (818688) and `n_embd` walk exactly. The two
settings differ only in the depth mix (maximum 1024 -> 768) and the rotation width.
`rk_hadamard_ok` admits 1024 and 512 at full width, and gives 768 only H_256 over three aligned
blocks. `KCHUNK`=768 measures `Dcal` = +0.36823, a move of −0.00258, under a third of the
smallest measured per-chunk step. At a held `C` and chunk count, the depth mix and the rotation
width have almost no effect on `Dcal`. The 832 setting's +0.021 therefore belongs to that walk
rather than to re-cutting in general.

`C` does not set the control's perplexity level. The same pair moves the control's PPL[4] by
0.0496, 15.1990 against 15.2486, on two runs identical in `C`, chunk count, population and
`n_embd` walk. Perplexity is deterministic here, so the gap is the two runs' difference and
nothing else. Three fits to the control level have worst in-sample residuals of 0.0255 (the
chunk count), 0.0295 (the maximum depth) and 0.0262 (the rotation width). The walk-specific
variation at a single point of `C` is 1.7-1.9x each of them.

The two channels dissociate again in the same two runs. Over the two calibrating windows the
controls differ by only −0.00264 nats, 1.02x what `Dcal` moves, so the whole 0.0496 lies in the
non-calibrating windows.

`D_res` measures +0.00732 at 832 and +0.01911 at 768. The runs with both families split span
[−0.00783, +0.01911], and the three runs with one family split or neither span [+0.02164,
+0.02376]. The collapse with both families split holds as a strict ordering over ten runs, with
a margin of 0.0025.

#### `BOOTMARGIN` and the no-cut response

`BOOTMARGIN` is headroom for the drift between the two calibrating forwards. With no row cut, a
margin of 1.00 leaves 51.911% of the columns clipping and 1.25 leaves 0.257%
[HW sweep 2026-08-22]. The no-cut response is non-monotone in the margin. Its minimum over the
four measured margins is at 1.25, not at the library default of 1.50.

**Conditions:** RK3576, Qwen2.5-1.5B, `-ub` 2048, `--chunks` 4. There is no row cut, so each
run's whole `Dcal` is the margin's effect. `Dcal` is the damage over the calibrating windows 1-2.

| `BOOTMARGIN` | `Dcal` (windows 1-2) | clipped of 1852160 | what the run measures |
|---|---|---|---|
| 1.00 | **+0.04066** | 961471 (**51.911%**) | clipping-dominated, not a resolution reading |
| 1.25 | **-0.00761** | 4760 (0.257%) | near zero confound |
| 1.50 | 0 | n/a | the control, zero by construction |
| 2.25 | +0.00811 | 0 | zero confound |

At no cut, `est1` is pass one's maximum over *all* `M` rows. Pass one's quantization changes
every downstream activation, so pass two's per-column maximum differs from pass one's. At zero
headroom, 51.9% of columns exceed `est1`. Margins of 1.50 and 2.25 cover essentially every
column.

A no-cut run isolates resolution only where the margin covers the drift. No zero-confound
resolution run therefore exists below margin 1.5, and `D_res(1.00)` is not measurable this way.

At margin 1.25, `Dcal` = -0.00761, a gain at full rows on this model. The reading covers the
calibrating windows only. It is not a converged-window result, and it does not change the
recommended configuration.

> [!NOTE]
> The `D_res(m)` forms used in the earlier sections have the wrong shape below the control
> margin, not only the wrong coefficient. The form linear in the quantum and the form quadratic
> in it are both monotone through 1.5. At 1.25 they give -0.00270 and -0.00198 against a
> measured -0.00761 (2.82x and 3.84x, right sign). At 1.00 they have the wrong sign, against
> +0.04066. The correction inside every published `R` and every `h`, `res_at_1p0`'s
> `-2/3 * D_res`, is wrong in sign at margin 1.00, the one margin it was written for.

Four of the five margins at cut `M`/4 have a measured no-cut run, so
`D_clip(cut, m) = Dcal(cut, m) - Dcal(no cut, m)` is measured minus measured there. Over matched
intervals the subtraction takes the margin exponent's local spread from 1.322x to 1.192x. It
tightens the axis toward a power law without straightening it. Only margin 1.875, a held-out
point, stays modeled.

#### The row cut at `M`/2

On Qwen2.5-1.5B at `-ub` 2048, a row cut to `M`/2 (`CALROWS`=1024) is a net accuracy gain at
both margins measured [HW sweep]. The damage is non-monotone in the cut, with a zero crossing
between `M`/2 and `M`/4. The separable model `D_clip = g(cut)*h(margin)` does not extend to
`M`/2.

**Conditions:** RK3576, Qwen2.5-1.5B, `-ub` 2048, scored against `q_ctl` at margins 1.5 and
2.25.

`CALROWS`=1024 costs `Dcal` = -0.00703 at margin 1.5 and -0.01065 at 2.25, *negative* at both.
The 8-window perplexity changes -0.382% and -0.571% (7.2257 and 7.2120 against a control
7.2534). The converged windows 3-8 measure -0.00393 +/- 0.00268 and -0.00586 +/- 0.00340 se,
with 4 of 6 better in each. With n=6, the converged column does not resolve either on its own.

As the cut deepens, `Dcal` runs:

| cut | 2048 | 1024 | 512 | 128 | 32 |
|---|---|---|---|---|---|
| margin 1.5 | 0 | -0.0070 | +0.1276 | +0.6761 | +1.9985 |
| margin 2.25 | +0.0081 | -0.0107 | +0.0266 | +0.1052 | |

The damage crosses zero between `M`/2 and `M`/4 at both margins. The `M`/4 cut lies past an
optimum, rather than at the low-cost end of a rising cost.

A row cut acts as an accidental percentile-clipping calibrator [hypothesis]. The sign, the
margin ordering and the crossing are measured, and the mechanism is not isolated. Pass one's
`est1` is a per-column maximum, and cutting its rows makes that maximum an under-estimate.
`sc2 = 127/(est1*margin)` then gets finer on every column while a few columns clip. Below some
cut, the finer quantum on the bulk is worth more than the clipped tail.

The gain is larger at the coarser margin, -0.0107 at 2.25 against -0.0070 at 1.5. On that
mechanism, the baseline quantum at 1.5 is already fine, which leaves less to gain [hypothesis].
A percentile or MSE-optimal clipping calibrator makes the same trade deliberately.

> [!NOTE]
> The cut factor `g` of the separable model cannot be a smooth positive function of the cut. It
> holds only at `M`/4 and deeper, and must not be extrapolated to shallower cuts.

The `D_res` model `D_res = 0.0081*(Q-1)/0.5`, with the resolution index `Q = (margin/1.5)*r`,
does not hold for `Q < 1.5`. Two runs that clip essentially nothing measure +0.0081 at Q = 1.50 and -0.0107 at
Q = 1.42. `D_res` is therefore not monotone in `Q`, and the margin and the cut do not reach the
returned output through one combined index.

The margin rescales `est1` by a constant on every column, and a row cut rescales it by a
column-dependent amount. Every `D_clip` reported in the sections above inherits that, so each
is `Dcal` minus a *model*, not a measured split.

#### Width of the `M`/2 gain

The `M`/2 gain is narrow. At margin 2.25, a run on each side of `M`/2 measures a positive `Dcal`
[HW sweep 2026-08-22]. A cut to 768 rows is already worse than no cut.

**Conditions:** RK3576, Qwen2.5-1.5B, `-ub` 2048, `BOOTMARGIN`=2.25.

**Not characterized:** the width of the minimum at margin 1.5, which has no 1536 or 768 run.

The whole cut axis at margin 2.25, as `Dcal` against `q_ctl`:

| cut | 2048 | **1536** | 1024 | **768** | 512 | 128 |
|---|---|---|---|---|---|---|
| `M`/ | 1 | 1.33 | 2 | 2.67 | 4 | 16 |
| `d` = 1-`r` | 0 | 0.020650 | 0.053643 | 0.090150 | 0.128489 | 0.255830 |
| `Dcal` | +0.00811 | **+0.00335** | -0.01065 | **+0.00931** | +0.02656 | +0.10521 |
| vs no cut | 0 | -0.00476 | -0.01876 | +0.00120 | +0.01845 | +0.09710 |

The minimum is at `M`/2, and the region around it is narrow. One step shallower (a 25% cut
rather than 50%) keeps only 25.4% of the gain. One step deeper (62.5%) keeps -6.4% of it, so 768
rows is worse than not cutting at all. The 8-window perplexities, which are not converged costs,
are 7.2534 at 2048, 7.2404 at 1536, 7.2120 at 1024 and 7.2648 at 768.

> [!NOTE]
> A row-cut curve on this axis must be measured at the cut it is quoted at. Four knots an octave
> apart do not resolve the minimum at `M`/2. Every interpolation tried over them puts the 1536
> cut on the wrong side of zero.

The mixed-`M` population is larger than its weight count suggests, and it does not explain the
minimum. Under `llama-perplexity` at `-ub` 2048, 352000 of 1852160 calibrated columns (19.0%)
sit at `M`=1024. By weight count that group is only 4 of 197. The prompt of `llama-bench` gives
a uniform `M`=2048, which is why `r` above is the `M`=2048 population's.

`CALROWS` clamps only when it is strictly below `M`, so that group is uncut at both 1536 and
1024, and first cut at 768. The mixture therefore predicts a kink at 768 and nothing at 1024,
while the minimum is at 1024. Re-weighting `d` by the mixture (0.0167 at 1536, 0.0435 at 1024,
0.0769 at 768 and 0.1143 at 512) rescales the abscissa and leaves the minimum exactly as sharp.
The minimum is not an artifact of the mixture.

The gain belongs to Qwen2.5-1.5B, not to the path. SmolLM2-1.7B at the same cut and margin
measures `Dcal` = +0.03770, where Qwen measures -0.01065: the opposite sign at 3.5x the
magnitude. Its 8-window perplexity is 7.4427 against 7.4265 (+0.218%), where Qwen's change is
-0.571% [HW sweep 2026-08-22]. The SmolLM2 runs use their own `-c` 2048 control. A `-c` 2048
run re-segments the corpus, so no `-c` 512 level and no other model's control is comparable.
The next section, §"The cut axis on SmolLM2-1.7B", completes that model's cut axis.

`CALROWS`=1024 is not a safe default at `-ub` 2048. It is the only setting on this axis that
beats the control at margin 2.25. Its advantage does not survive a 25% move of the
`CALROWS`/`M` ratio in either direction. On the second model run at this ubatch it is a cost
rather than a gain. `CALROWS` is an absolute row count, so every ubatch change moves that ratio.
Two models make up the whole corpus at `-ub` 2048, so quote the `M`/2 gain as Qwen2.5-1.5B's
number and nothing more.

#### The cut axis on SmolLM2-1.7B

SmolLM2-1.7B has no row-cut optimum: its damage is monotone in the cut depth, and every cut is
worse than not cutting [HW sweep 2026-08-22]. Qwen2.5-1.5B's minimum at `M`/2 is that model's
alone, and no cut is free or better on both models at `-ub` 2048. The resolution channel
differs 2.54x between the two models.

**Conditions:** RK3576, `BOOTMARGIN`=2.25, `-ub` 2048. Each SmolLM2-1.7B run is scored against
its own `-c` 2048 control (PPL[8] 7.4265).

| cut | `M`/ | Qwen `Dcal` | Qwen ppl8 | SmolLM2 `Dcal` | SmolLM2 ppl8 | SmolLM2 clip |
|---|---|---|---|---|---|---|
| 2048 | 1 (none) | +0.00811 | -0.290% | **+0.02060** | -0.606% | **0.0000%** |
| 1536 | 1.33 | +0.00335 | -0.179% | **+0.02405** | +0.191% | 0.0294% |
| 1024 | 2 | -0.01065 | -0.571% | +0.03770 | +0.218% | n/a |
| 512 | 4 | +0.02656 | +0.301% | **+0.05278** | +0.063% | 0.3705% |

Subtracting each model's own no-cut level leaves the row cut's own term. That term has opposite
signs on the two models at both shallow cuts: -0.00477 against +0.00345 at 1536, and -0.01876
against +0.01710 at `M`/2. At `M`/4 it costs on both models, +0.01845 against +0.03218.

The no-cut run clips exactly zero columns on both models (0 of 1500160 and 0 of 1769472). It
therefore measures the resolution channel with its confound nulled, not merely small. It costs
+0.00811 against +0.02060, a 2.54x model difference. A third model measures +0.01626 at the same
zero confound (0 of 1272320 columns), between the other two. The channel therefore spans 2.54x
across three models [host-computed]. It does not follow the margin index, which two of the three
share to 0.9%.

Splitting the models' gap at the one cut both ran gives 25.8% to the margin channel and 74.2% to
the cut channel. Both channels differ between the models, and the cut channel accounts for three
quarters of the gap.

The mixed-`M` population differs between the models, so remove it before quoting a magnitude.
Read off each model's own control CALMAP, the `M`=1024 minority is:

| model | `M`=1024 columns | calibrated columns | share |
|---|---|---|---|
| Qwen | 352000 | 1852160 | 19.0% |
| SmolLM2-1.7B | 151552 | 1769472 | 8.57% |
| Llama-3.2-1B | 309760 | 1272320 | 24.35% |
| SmolLM2-360M | 110464 | 651264 | 16.96% |

The shares spread 2.8x, and the spread does not follow width: the two models sharing
`n_embd`/`n_ff` sit at its two ends. `CALROWS` clamps only when it is strictly below `M`, so
more of Qwen's columns escape uncut at a given cut.

##### The mixed-`M` population

The mixed-`M` population is the output head plus one block's FFN, and the GGUF header gives it
exactly [host-computed]. Every block's q/k/v/o sits at `M`=2048, and the `M`=1024 rows are the
output head plus exactly one block's gate, up and down. The mechanism is upstream of the NPU
path. `llama-perplexity` asks for logits on only half of each 2048-token window, so ggml
prunes the graph to those output rows before the final block's FFN. Everything before the prune
runs at the full ubatch.
On Qwen2.5-1.5B the four weights at `M`=1024 are `blk.27.ffn_{down,gate,up}` and `token_embd`,
the last block's FFN and the output head [HW sweep 2026-08-21].

**Conditions:** `llama-perplexity` at `-ub` 2048. Under `llama-bench` the `M` column is uniform.

**Not characterized:** which block the FFN belongs to on the other four models. There it is
inferred from the prune rather than read from the `key` column.

The CALMAP logs one column per output column of each offloaded GEMM. A K-split `ffn_down` logs
its `N` columns once per chunk, so per calibrating call:

    per_block = 2*n_embd + 2*head_kv*head_dim + 2*n_ff + n_embd*chunks(n_ff)
    total     = n_blocks*per_block + vocab
    minority  = vocab + 2*n_ff + n_embd*chunks(n_ff)

where `chunks(K)` is `rk76_ksplit`'s greedy walk down `RK76_KCHUNK_PREF`, and is 1 for any `K`
the RK3576 takes whole. The formula reproduces all ten measured integers, five minority counts
and five totals, over five models, exactly. Both second-order terms are required. The minority
share each reduced formula gives, against the measured share:

| Measured | Head alone | Per-chunk `ffn_down` logging dropped |
|---:|---:|---:|
| 19.005% | 16.41% | 21.50% |
| 8.565% | 5.56% | 9.82% |
| 24.346% | 20.16% | 29.04% |
| 16.961% | 15.09% | 16.96% |

SmolLM2-135M (576/1536, 30 blocks, `head_kv` 3, vocab 49152) has a total of 409344 and a
minority of 105600, a share of 25.797%. That share is the highest of the five, which run
8.565 / 16.961 / 19.005 / 24.346 / 25.797%. The smallest model sits highest, because the
dominant term is `vocab/total` and a smaller model divides the same vocabulary by less.

The share is therefore set by the vocabulary size, the FFN width and the `ffn_down` chunk
count. It follows neither width nor depth, because its dominant term is `vocab/total` and the
vocabulary is an axis independent of both. The two models sharing `n_embd`/`n_ff` (vocab 49152
and 128256) sit at opposite ends of the spread for that reason. Compute the share from the
header before quoting any cross-model calibration magnitude. It needs no CALMAP pass.

One model treats the damage as a function of the per-column ratio alone, and adds across the
two groups. It takes the `M`/4 fraction's model ratio from a raw 1.74x to 1.19x [expected]. The
figure is derived, not measured: it is a linearization, and no independent consistency check
exists inside the data. The re-weighting cannot change the opposite signs. Re-weighting a
population multiplies the cut term by a positive factor, and a positive factor cannot change a
sign.

##### `Dcal` against the 8-window perplexity

`Dcal` and the 8-window perplexity disagree in direction across the two models, which is what
the calibrating-window scope means in practice. At `M`/4, SmolLM2-1.7B costs 1.99x Qwen's
`Dcal`, while its 8-window change is 5x smaller (+0.063% against +0.301%). The margin alone at
full rows improves the 8-window perplexity on both models (-0.290% and -0.606%), while it costs
on the calibrating windows. `BOOTMARGIN`=2.25 therefore does not trade accuracy away outside the
windows it repairs.

The converged column (windows 3-8) is negative in 9 of the 10 runs either model has here. At n=6
per run and with a shared control, those are not independent readings, and none resolves.

**Not characterized:** any margin but 2.25, any ubatch but 2048, decode, a third model, and
SmolLM2-1.7B between `M`/2 and `M`/4, where Qwen's zero crossing lies. A converged cost is not
characterized either, because `Dcal` is the two calibrating windows' damage. A positive `Dcal`
is not a converged cost until a run at depth measures it.

Heterogeneity between the models at `-ub` 512 is Cochran's Q = 6.665 on 2 df, which keeps a third model
outside this comparison. The comparison shows where the models differ, in a cut-indexed term or
not, and not why. A cause indexed by neither would appear in the no-cut run identically, and
the two models differ in `n_embd`/`n_ff` (1536/8960 against 2048/8192).

##### Clip-fraction statistics

No single statistic of the clipping predicts the damage [HW sweep 2026-08-22]. A saturating form
in the clip fraction fails across the runs with a row cut, and so do the relative severity and
the clipped mass. The damage comes through clipping, but no function of the clip
fraction alone predicts it. The saturated columns of a run with a row cut always sit about a
tenth over the ceiling, and only their number moves.

**Conditions:** RK3576, Qwen2.5-1.5B, `-ub` 2048.

The `llama-bench` prompt under-states the clip fraction 2.0-5.5x against the wiki corpus. Every
clip fraction in these sections comes from `llama-bench`, whose prompt is synthetic, and every
`Dcal` comes from the corpus. The clip fractions at the one cut where both prompts have a
CALMAP, and at `M`/4:

| Cut | Margin | Corpus | `llama-bench` | Corpus / bench |
|---|---:|---:|---:|---:|
| The shared cut | 1.5 | 2.7975% | 1.3775% | 2.03x |
| The shared cut | 2.25 | 0.0968% | 0.0177% | 5.47x |
| `M`/4 | 1.875 | 1.6602% | 0.4963% | 3.35x |

The undershoot `r` transfers far better between the prompts: 0.94335 corpus against 0.94636
bench at `M`/2 (0.32%), and 0.90584 against 0.87151 at `M`/4 (3.9%). Quote a clip fraction with
its prompt named, and treat a `llama-bench` figure as a lower bound. The undershoot `r` can be
transferred. The fitted model above is indexed by cut and margin rather than by the clip
fraction, so the prompt difference does not affect it.

`AROW` does not move the clip fraction by a constant factor, and its effect does not keep its
sign. The factor by which the flag moves each run's clip fraction:

| Run | Factor |
|---|---:|
| No-cut control | 8.6x (0.2600% -> 0.0303%) |
| `q_r512` | 1.85x |
| `q_shipfrac` | 3.45x |
| `q_r128` | 1.15x |
| `q_r32` | 0.923x (35.4030% -> 38.3480%) |

At `q_r32` the flag increases clipping. A single regime factor cannot transfer a clip fraction
across the cut axis. One fitted on the shallow cuts extrapolates through 1 before the deep ones.

The clip fraction at any margin is computable off-line from a fixed cut's bytes. Pass one's
`est1` is margin-free given its input. A column therefore clips exactly when the oracle's
uncapped colmax exceeds `est1*margin`, and both vectors are recorded for every cut already run.
Against the device the computation measures 1.6% high at margin 1.5, 1.7% low at 2.25 and 5.7%
high at 1.875.

**Not characterized:** the prompt difference beside `AROW`. The runs the off-line computation is
checked against are `llama-bench` runs, whose prompt is synthetic, while every `Dcal` comes from
the wiki corpus.

##### The `est2` readback and `Dcal` reconstruction

`est2` is a readback, not the accumulator [HW sweep 2026-08-22]. Pass two writes
`est2[n] = max(1, max_m |C8|) / sc2[n]` with `sc2 = 127/(est1*margin)`, and sets `sat[n]` at
code 127 (`rk76_bootstrap_pass()` in `ggml-rocket.cpp`). On a saturated column `est2` is
therefore capped at the ceiling. It is quantized to the returned output's own code, margin/127
of `est1`, which is 1.77% at margin 2.25, with a minimum of one code.

A no-cut run at margin 2.25 saturates nothing. Its `est2` is therefore the only uncapped
per-column maximum on record, and the one measurement that can quantify severity.

`Dcal` is reconstructible from any run's own window line. A `llama-perplexity` `[k]` value is
the cumulative PPL over the first k windows. Window k's nats are therefore
`k*ln(P[k]) - (k-1)*ln(P[k-1])`, and `Dcal` is the sum, configuration minus control, over the
two calibrating windows. The reconstruction reproduces four independently scored values to
< 5e-4 nats.

#### The row cut's saving

Below 128 rows, a calibrating call's submit count stops falling, and its time still falls with
the rows. The runs `r128` and `r64` (`CALROWS` 128 and 64) emit the identical 9710 submits, yet
`calibrate` still falls 86.19 -> 82.23 ms, 13x the repeat spread [HW sweep 2026-08-21]. The
submit count likely follows a row tile quantized at 128, while device time inside a submit
follows the actual rows [hypothesis].

The row cut's saving is linear in the rows the cut removes, so it is a property of `M` and not of
the setting alone. It is 18% at `-ub` 512, where `CALROWS`=128 with `BOOTMARGIN`=2.25 removes
17.93% of `calibrate` per calibrating call (105.02 -> 86.19 ms) [HW sweep 2026-08-21]. The saving
is 7.42 s once per process, over 394 calibrating calls. With the common-mode process shift that its
own `entry` bucket reports removed, `BOOTMARGIN`=2.25 alone costs nothing, at -0.17%. Its submit
count is identical to the control's.

**Conditions:** H96 MAX M9, `M`=512, `NCAL` at its default of 2 (the accuracy runs'
configuration), CALMAP off, `--chunks 4`, `setarch -R`, one `.so`, corpus and model.

**Not characterized:** decode, another model, the BO pool on, an unpinned layout, and the
accuracy of any row count but 128.

| `CALROWS` | `calibrate`/calibrating call | saving | submits |
|---|---|---|---|
| full `M`=512 | 105.02 ms | n/a | 12828 |
| 256 | 93.21 ms | 11.24% | 10868 |
| **128** (`M`/4) | **86.19 ms** | **17.93%** | **9710** |
| 64 | 82.23 ms | 21.70% | **9710** |

Fitting the three cut points through the origin gives `s(M,M1) = (M - M1) / (1099 + 2M)` in rows.
Fitted entirely at `M`=512, it gives 29.6% and 37.0% for the two `M`=2048 figures. The measured
values are 28.2% (`CALROWS`=512) and 36.3% (`CALROWS`=128). The formula is within 1.4 points across
a 4x change in `M`, a different instrument state and a different epoch. The saving therefore scales
with the ubatch: ~18% at `-ub` 512, and ~37% at 2048 and ~43% at 4096 from the formula [expected].

As `CALROWS` approaches 0, the saving at `-ub` 512 approaches 24.1%. A cut at `M`/4 already takes
74.3% of it, against a recorded +19.15% perplexity cost at `M`/16. `M`/4 is therefore
near-optimal on cost as well as on accuracy.

Above 128 rows the saving is proportional to the submits removed, to 0.2%, at 6.025 / 6.037 /
5.941 us per removed submit on three runs. Below 128 rows a second term provides the saving.

> [!NOTE]
> Time a row-count sweep, and do not read its saving off the submit count. Below 128 rows the
> submit count is flat while the saving continues, by 4 ms a call.

`NCAL` does not change this fraction. It changes how many calibrating calls a process makes, not
what one costs. The default `NCAL`=2 measures -17.9% here, against -17.2% measured at `NCAL`=1
two epochs earlier (86.1 -> 71.4 ms a calibrating call, submits 10690 -> 9131). The gap between
the recorded 28.2 and 17.2 figures is therefore `M`, not `NCAL`. The 4-window aggregates in these
runs (8.7163 / 8.7435 / 9.0511) are the calibration-window artifact described earlier, and are
not accuracy measurements.

### Calibration with `NCAL`=1

With `ROCKET_RK3576_NCAL`=1, each weight takes one calibrating forward instead of two. Pooled
over three models, the converged-window perplexity cost of the change is +0.566%, 2.83 se from
zero [HW sweep 2026-08-20] (§"Llama-3.2-1B-Instruct"). On Qwen2.5-1.5B and on SmolLM2-1.7B alone
it is not resolvable from zero [HW sweep 2026-08-12], and Llama-3.2-1B-Instruct resolves it at
3.0 se. The change saves 58.6 ms per replaced call, once per process.

**Conditions:** H96 MAX M9, `AROW`=1, K-split on, wikitext-2, `-ub` 512. Each `NCAL`=1 test
point is paired window by window against a same-build `NCAL`=2 control.

**Not characterized:** Qwen2.5-1.5B over every window of the text, so its 8-window +0.36% is
not resolved. Decode, another `-ub`, and `CALROWS`'s -28.2% saving at `NCAL`=1, which is
[expected]. Whether the 4.2x factor below survives the per-row scale on any other term: its
test point, `AROW` against `nosplit`, is an 8-window difference.

Dropping the second calibration forward changes what the frozen scale is built from. Its cost
therefore appears in the converged windows, which no margin covers. The row cut's cost,
by contrast, falls in the calibrating window.

On Qwen2.5-1.5B at `--chunks 8`, against the same-build `NCAL`=2 control of 9.9348:

| Qwen2.5-1.5B, `AROW`=1, K-split on | PPL | paired d(NLL) | windows 3-8 |
|---|---:|---:|---:|
| `NCAL`=1 | 9.9702 | +0.0036 +/- 0.0039 (se) | +0.0028 +/- 0.0050, 4 of 6 positive |
| `NCAL`=1 + `CALROWS`=128 / `BOOTMARGIN`=2.25 | 9.9501 | +0.0015 +/- 0.0071 | -0.0055 +/- 0.0071 |

Neither difference is resolvable from zero. The library's saturation report, which is live in
the converged phase, stayed silent in both test points. The converged saturating fraction is
therefore under its 0.1% threshold.

SmolLM2-1.7B is the model whose `ffn_down` is `K`=8192. It costs 4.2x more than Qwen for a
frozen per-tensor scale (6.77x against 1.600x). Its `NCAL`=1 cost was measured at two window
counts, with the same `.so` in the same boot:

| SmolLM2-1.7B, `AROW`=1, K-split on | windows | `ctl` (`NCAL`=2) | `NCAL`=1 | converged windows |
|---|---:|---:|---:|---:|
| `--chunks 8` | 8 | 10.3354 | 10.4800 (**+1.40%**) | +0.0148 +/- 0.0113, 3 of 6 positive |
| every window the text has | 29 | 8.5473 | 8.5735 (**+0.31%**) | **+0.0024 +/- 0.0044**, 13 of 27 |

At 29 windows the converged cost is 0.55 se from zero, and within 0.06 se of Qwen's +0.0028.
The two models agree on this term, and the 4.2x factor does not appear in it.

> [!NOTE]
> Eight windows of wikitext-2 do not resolve a converged-window cost on this path. On
> SmolLM2-1.7B the 8-window +1.40% is +0.31% over all 29 windows, a 4.5x error set by which
> windows the text starts with. The per-window values are deterministic, so the uncertainty is
> the window sample alone. Score such a cost over every window the text has.

The row cut behaves the same way on SmolLM2-1.7B. Its whole cost is again the one
calibrating window: +0.0241 nats at window 1, against Qwen's +0.0296 at the same window, with
the same sign. Over windows 2-8 it is -0.0004 +/- 0.0040, independent of `NCAL`, which excludes
composition at 5 se. It was measured composed with `NCAL`=1. `CALROWS` alone at the default
`NCAL`=2 on this model is not characterized.

The two settings are independent. A second forward can repair a column whose row-cut
first-pass estimate under-read it, so the row cut's frozen-scale damage could have composed
with `NCAL`=1. On Qwen2.5-1.5B it does not. The composed test point differs from plain `NCAL`=1
over windows 2-8 by -0.0065 +/- 0.0074 (se). The row cut's calibration-window cost is
unchanged: window 1 is +0.0296 nats here and +0.0296 at `NCAL`=2.

#### Saving from `NCAL`=1

A calibrating call costs about twice a converged one, measured per frontend call through the
profile buckets [HW sweep 2026-08-12]:

| per frontend call, `-ub` 512 | `NCAL`=2 | `NCAL`=1 |
|---|---:|---:|
| calibrating (`calibrate`/`cal_calls`) | 117.01 ms | 115.38 ms |
| converged (`entry`/converged calls) | 58.14 ms | 57.09 ms |

The ratio is 2.01x, one bootstrap pass per extra device pass, as the code emits. `NCAL`=1
replaces one calibrating call per weight with a converged one. Its saving is therefore
`calibrate`-per-call minus `entry`-per-call, not the `calibrate` bucket. The net 58.6 ms per
replaced call, over the model's 197 offloaded GEMM sites, projects 11.5 s. The measured
offloaded-path change is 11.6 s (90.2 -> 78.6 s).

The saving is incurred once per process and does not scale with prompt length. The 11.6 s
(-12.9%) is a share of a four-window run's offloaded path, not of a deployment.

### The calibration regime

A calibrating call costs about twice a converged one, and a short bench never leaves the
calibration regime. `ROCKET_RK3576_NCAL` defaults to 2, and a calibration forward is two device
passes (the colmax bootstrap). The first two micro-batches each weight sees therefore cost about
twice what every later one does. Per frontend call at `M`=512, with the row-major calibration
scan, a calibrating call costs 2.01x a converged one [HW sweep 2026-08-12] (§"Saving from
`NCAL`=1").

The removable cost of calibration is the `calibrate` bucket minus the `entry` it displaces. In
the column-outer build's measurement of 2026-08-11 at `pp2048 -ub 2048` (§"Calibration cost in
the W8A8 frontend"), `calibrate - entry` = 171.3 - 47.5 = 123.8 s a process. A calibrating call
returns before the converged path's device call, so it never enters the `entry` bucket. The
directly observed excess of the two calibrating forwards over a converged one is 127.8 s, a 3.2%
residual. The residual is the calibrating path's own `dequant` and the scale loops. The 171.3 s
bucket overstates the saving by 1.39x, and any bucket standing in for a branch needs the same
subtraction.

#### Calibration share of a `llama-bench` run

`llama-bench` runs one untimed warmup and then `-r` timed reps. The share of a bench in the
calibration regime is therefore set by how many micro-batches a pass contains, `-p / -ub`. At
`-p 2048`:

| `-ub` | forwards per weight, `-r 1` | of which calibration |
|---|---:|---:|
| 512 | 8 | 2 (25%) |
| 1024 | 4 | 2 (50%) |
| 2048 | 2 | **2 (100%)** |

A `-ub` sweep at `-r 1` therefore varies the micro-batch and the calibration share together. Its
reported t/s falls 12.07 -> 11.49 -> 9.67, in the direction that makes the larger micro-batch
look worse.

> [!NOTE]
> A `pp512 -r 1` bench of this path is 100% calibration regime: every offloaded matmul runs the
> device twice. Read the `entry` and `calibrate` buckets, which separate the two regimes, rather
> than the wall-clock time. Alternatively, give the run enough reps that the converged path
> dominates.

#### Bucket accounting

The profile buckets account for both paths. On the converged path, the frontend's `entry`
agrees with the library's own accounted total for the same calls to 2.2-5.0%. On the
calibration path, the bootstrap's per-column maximum scan has its own `calscan` bucket, at 1.0%
of `calibrate`.

One-time weight preparation is 26.7-29.7 s over 166 weights: `wt_rotate` 15.6-17.1 s and
`wt_quant` 11.0-12.6 s. It is incurred once at first use, and cached by the weight's name plus a
content fingerprint. It therefore falls in `llama-bench`'s untimed warmup and in a user's first
prompt.

> [!NOTE]
> Build `ggml-rocket` for the RK3576 with `-DGGML_ROCKET_NATIVE_FP16=OFF`. The option defaults
> on and adds `-march=armv8.2-a+fp16` for the RK3588's A76 and A55 cores. The RK3576's CPU is
> ARMv8.0-A (`/proc/cpuinfo` has neither `asimdhp` nor `asimddp`). A `.so` built with the option
> on would raise SIGILL in the convert kernels.

## Accuracy of the W8A8 path inside a model

The W8A8 path costs 1.011x perplexity, measured on the RK3576 inside a model
[HW sweep 2026-08-11]. It runs the same perplexity pass 1.99x faster than the CPU.

**Conditions:** H96 MAX M9, llama.cpp b10356, Qwen2.5-1.5B F16. `llama-perplexity` over the first
8 chunks of wikitext-2 test at `n_ctx`=512. Both configurations use the same GGUF, the same text
and the same boot. The NPU path runs the default bootstrap at `CALSAFE`=3.0.

| configuration | wikitext-2 PPL (8 chunks) | wall-clock time |
|---|---:|---:|
| CPU only, F16 | 9.7872 +- 0.58866 | 13.01 min |
| NPU, W8A8 | 9.8969 +- 0.59556 | 6.55 min |

The wall-clock ratio, 1.99x, agrees with the `pp512` bench's 2.01x on the same denominator. The
1.008x and 1.020x figures are host arithmetic over a simulator. The 1.011x is measured on the
device inside a model.

The figure scores the block-diagonal rotation. Every offloaded call here is `K`=1536, which is
neither a power of two nor `60*2^k`, so every call takes the block-diagonal Hadamard fallback.
The fallback composes with the rest of the path. Without rotation, a 2% per-column divisor
change took SmolLM2 from 10.58 to 2.6e10. Against that, a 1.1% change is small.

> [!NOTE]
> The 1.011x and the simulator's 1.008x are different quantities, so do not take their
> difference. The simulator's figure is against fp32 with an exactly frozen colmax. The 1.011x
> is against the same GGUF's F16 CPU run, with the default bootstrap at `CALSAFE`=3.0, over 8
> chunks whose own error bar is +-0.59. It shows that no part of the composed device path costs a
> model an order of magnitude [HW sweep 2026-08-11].

## Output scale of the int8 entry

The RK3576 DPU's integer epilogue is `(acc + A[oc]) * C[oc]`, then one `(v*MUL)>>SHIFT` per
task. A per-output-channel output scale is therefore available on chip, as an integer `C` ramp
at a resolution of `0.5/C[oc]` with one shift per task.

With one per-tensor output scale writing int8, the int8 GEMM entry raises a real LLM's prefill
perplexity 1.11x to 2.26x, depending on the model. A Hadamard rotation together with a
per-output-channel requant takes both models measured to 1.00x. Unlike the datapath limits in
[../chips/rk3576.md](../chips/rk3576.md#datapath-blockers-for-a-gemm-frontend), which the entry
refuses, this loss is silent.

**Conditions:** host simulation of the entry's expression,
`sat8(round_even((A*B^T + bias) * scale))`, over each model's own prefill activations, 512
wikitext-2 tokens. SmolLM2-1.7B (`d`=2048, ffn=8192, 24 layers, 144 offloadable projections)
and Qwen2.5-1.5B (`d`=1536, ffn=8960, 28 layers, 168).

Every scale is an oracle taken from the tensor it quantizes, so these figures are the entry's
best case. A projection is offloadable exactly where the entry accepts it (`K`%32, `N`%32,
`K` < 6176). Every `d_model`-input projection is then on the NPU, and the FFN down-projection
stays on the CPU.

Per GEMM, relative Frobenius error against a float64 reference, over 36 offloadable test
points:

| what quantizes how | SmolLM2-1.7B | Qwen2.5-1.5B |
|---|---:|---:|
| fp16, the error level the CPU path already delivers | **0.0003** | 0.0003 |
| per-tensor A, per-tensor B, per-tensor int8 out | 0.0965 | 0.0712 |
| per-row A, per-channel B, per-tensor int8 out | 0.0439 | 0.0408 |
| Hadamard-rotated A and B, per-tensor, int8 out | 0.0202 | - (`K`=1536 is not 2^k) |

The end-to-end table replaces every offloadable projection in every layer. It scores
wikitext-2 perplexity over 8 windows of 512 tokens, and top-1 agreement against the fp32 run:

| configuration | SmolLM2 ppl | x fp32 | top-1 | Qwen ppl | x fp32 | top-1 |
|---|---:|---:|---:|---:|---:|---:|
| fp32 / fp16 | 11.1937 / 11.1924 | 1.000 | 99.73% | 12.8403 / 12.8385 | 1.000 | 99.63% |
| the entry as specified (per-tensor everywhere) | 25.2618 | **2.26x** | 61.99% | 14.2294 | **1.11x** | 83.18% |
| per-row A + per-channel B through it | 964.3787 | **86x** | 60.69% | 13.2236 | **1.03x** | 91.19% |
| per-channel output requant, no rotation | 17.2697 | 1.54x | 72.36% | 13.6248 | 1.06x | 85.86% |
| Hadamard rotation, per-tensor output requant | 14.6771 | 1.31x | 76.78% | 13.1144 | 1.02x | 90.48% |
| per-axis inputs, no output requant at all | 15.1168 | 1.35x | 77.22% | 13.1369 | 1.02x | 93.51% |
| **Hadamard + per-channel output requant** | **11.2366** | **1.004x** | **97.29%** | **12.8645** | **1.002x** | **96.85%** |

The cost depends on the model by more than an order of magnitude. Per-axis input scales are
free at this interface, because the caller quantizes A and B and the entry's `scale` is the
output's alone. Per-row activation and per-channel weight scales lower the per-GEMM error on
both models. End to end they take Qwen from 1.11x to 1.03x and SmolLM2 from 2.26x to 86x.

SmolLM2's damage under per-axis inputs is to calibration rather than ranking. Its top-1
agreement is 60.69% against the per-tensor configuration's 61.99%, essentially unchanged,
while perplexity goes 25 -> 964. The mechanism is not characterized.

> [!NOTE]
> A per-GEMM error norm does not predict the end-to-end cost. A port must not assume
> per-axis input scales help, and must measure a candidate model end to end.

The rotation and the per-output-channel requant hold on both models, and neither is enough
alone. Alone, the rotation scores 1.31x on SmolLM2 and 1.02x on Qwen, and the requant 1.54x
and 1.06x. Together they score 1.004x and 1.002x at 97% top-1. The rotation runs on the host
and needs no library change: an orthonormal Hadamard along `K`, applied before quantization.
Qwen's `K`=1536 takes the weaker block-diagonal form over 512-element blocks and still reaches
1.002x.

The per-channel requant is a library change on hardware the DPU already has.
`rocket_conv2d_rk3576.c` programs the `C` ramp for per-axis convolution.
`rocket_matmul_int8_rk3576_perc()` takes a per-column output scale and plans it with the same
per-axis planner, one `(MUL, SHIFT)` per N tile. The per-tensor entry,
`rocket_matmul_int8_rk3576()`, pins `in_scale`/`w_scale` to 1 and puts the caller's factor in
`out_scale`.

The 1.004x is a ceiling for this path rather than its figure. The simulation gives every
output column an exact per-column scale, and the hardware gives the integer `C` ramp. The
ramp's resolution is why the per-axis ResNet-18 scores 89 of 100 against a per-tensor control's
93.

Every number in this section is host arithmetic. The integer ramp's own cost, 0.0006 of ratio on
Qwen2.5-1.5B, is in §"Sources of the per-column output scale". The composed device path's is in
§"Accuracy of the W8A8 path inside a model".
[host-computed, arithmetic over two real models, `M`=512, 2026-08-10]

### Host cost of the rotation and quantization

**Conditions:** H96 MAX M9, the A72 cores (`taskset -c 4-7`), `M=512 K=N=2048`, 2026-08-10.

The host work a W8A8 path adds leaves most of its speedup, because the down-projection
dominates. Per call the path adds an activation quantization and a rotation, which the
NPU-against-CPU comparison above did not include. Quantizing takes 1.08 ms, the FWHT 2.06 ms
and dequantizing 0.57 ms, in all 3.70 ms on four cores and 12.22 on one. The total is 13% of
the entry's 28.2 ms at that shape, and it lowers the shape's speedup from 4.72x to 4.18x.
[HW sweep, H96 MAX M9, 2026-08-10]

Weighted by a Llama-3.2-1B layer's MAC shares, the comparison's speedup moves only from 2.44x
to 2.37x. The 2.44x is the GEMM-only weighting of this measurement. An earlier weighting of the
GEMMs, on 2026-08-05, gave 2.39x
([../chips/rk3576.md](../chips/rk3576.md#the-int8-gemm-against-a-real-cpu-gemm-at-llm-prefill-shapes)).

The down-projection, which the entry refuses, is 65% of the NPU path's wall-clock time and adds none of the host work. The rotation is also per input, not per call. The q, k
and v projections share one, and gate and up share one, so a port rotates twice a layer rather
than six times. [HW sweep, H96 MAX M9, 2026-08-10] The down-projection's CPU time is taken as
gate/up's at equal MACs [expected].

## Sources of the per-column output scale

`rocket_matmul_int8_rk3576_perc()` takes each column's output scale, `scale_n`, from its
caller. Every perplexity figure above uses an oracle per-column scale, taken from the
accumulator it is about to quantize. A caller holds A and B and no accumulator, so the oracle
figures are the path's ceiling and not a frontend's. The table below scores what a caller can
supply.

**Conditions:** host simulation, Qwen2.5-1.5B, the same eight windows of 512 wikitext-2 tokens
and the same 168 offloaded projections.

| per-column scale source | ppl | x fp32 | top-1 vs fp32 |
|---|---:|---:|---:|
| oracle, exact per-column (`hadamard_chanout`) | 12.8645 | 1.0019 | 96.85% |
| oracle, through the shift-0 ramp | 12.8724 | 1.0025 | 96.63% |
| **frozen over 2 disjoint calibration windows** | **13.4451** | **1.047** | **87.28%** |
| the analytic bound `128*sum_k\|Bq[n][k]\|` | 389.5999 | 30.342 | 19.63% |
| the frozen scale's perturbations, dealt to the wrong columns | 12.9315 | 1.007 | 92.70% |

The scale source is the largest single term on this path. Its excess over the exact-scale
ceiling is 0.045 of ratio, against the integer ramp's 0.0006 on the same model, a factor of
seventy-five. An accuracy budget that counts the ramp and not the scale source is low by an
order of magnitude.

The frozen scale's damage is in the tail rather than the average. At the median it is 1.083x
looser than the call's own column maximum, and it saturates less of the output than the oracle
does (0.180% against 0.212%). Over 1344 forwards, the tightest single column sits at 0.228x,
4.4x too tight, and saturates hard. A maximum over the calibration set is therefore the wrong
estimator for the tail. A global safety factor and a larger calibration set are measured below
(§"A global safety factor on the frozen scale", §"The calibration window count, `NCAL`"). A
percentile and a per-column safety factor are not characterized.

The harm is the correlation rather than the magnitude. The last row deals the same per-column
perturbations to random columns, and scrambling which columns get the bad scale makes the
model six times better. A column the calibration set underestimated is a column whose
activations grew. Clipping that column costs, where the same clip on a random column is nearly
free.

The analytic bound fails on aim alone. It overshoots a real column's accumulator by 57.7x
median (28-266x), using about six bits of the int8 output on headroom no value reaches. It
saturates 0.000% of the output. Its C ramp is meanwhile nearly exact, with `worst_rel_err`
0.15% against the oracle's 0.59%. Under this source `cs/cmax` is one constant in every
unclipped column, so `C` sits exactly at its cap. Only the integer rounding of `C` is left.

The ramp's resolution and the scale's aim are independent sources of error, and on this path
the aim is the one that costs.
[host-computed, arithmetic, Qwen2.5-1.5B, 2026-08-10]

On SmolLM2-1.7B the scale source costs nearly three times as much, and the random-column
control splits differently.

**Conditions:** host simulation, SmolLM2-1.7B, its own eight windows and 144 offloaded
projections.

| per-column scale source | ppl | x fp32 | top-1 vs fp32 | tightest column | saturating |
|---|---:|---:|---:|---:|---:|
| oracle, exact per-column | 11.2366 | 1.0038 | 97.29% | 1.000x | 0.221% |
| oracle, through the shift-0 ramp | 11.2855 | 1.008 | 96.14% | 1.000x | 0.213% |
| **frozen over 2 calibration windows** | **12.6872** | **1.133** | 84.57% | **0.161x** | 0.176% |
| the analytic bound | 1.043e9 | 9.3e7 | 3.22% | 2.352x | 0.000% |
| the same perturbations, wrong columns | 12.0254 | 1.074 | 90.06% | 0.127x | 0.360% |

The excess over SmolLM2's own exact-scale ceiling is 0.129 of ratio, against Qwen's 0.045. The
scale source is the largest term on both models, but 2.9x larger on SmolLM2, so a port that
budgets from one model budgets low. SmolLM2's tail is deeper too (0.161x, 6.2x too tight,
against 0.228x).

The random-column control costs 0.070 of SmolLM2's 0.129, 54% of the damage, against 12% of
Qwen's 0.045. The correlation adds damage on both models, and on neither is it all of the
damage. On SmolLM2, about half of it is what any perturbation of that size costs. The analytic
bound is unusable on both models, and its SmolLM2 result (9.3e7x, 65.9x median overshoot) is a
destroyed model.

## A global safety factor on the frozen scale

A global safety factor on the frozen scale recovers 87% of the scale-source term, on both
models. The factor, `CALSAFE`, multiplies the frozen per-column maximum, the colmax, before it
is inverted. It trades output resolution in every column for clipping headroom in the tail.

**Conditions:** host simulation, the same windows and calibration set, `NCAL`=2 throughout. The
ratio and saturation columns are over 1344 forwards on Qwen and 1152 on SmolLM2.

| CALSAFE | Qwen ppl | x fp32 | ratio median | tightest | saturating | SmolLM2 ppl | x fp32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 13.4451 | 1.047 | 1.083x | 0.228x | 0.180% | 12.6872 | 1.133 |
| 1.25 | 13.0510 | 1.016 | 1.349x | 0.273x | 0.021% | 11.8534 | 1.059 |
| **1.5** | **12.9395** | **1.008** | 1.614x | 0.308x | 0.004% | - | - |
| 2.0 | 12.9510 | 1.009 | 2.156x | 0.410x | 0.000% | **11.4165** | **1.020** |
| 3.0 | 12.9452 | 1.008 | 3.233x | 0.588x | 0.000% | - | - |
| 4.5 | 13.0487 | 1.016 | 4.846x | 0.889x | 0.000% | - | - |

Perplexity against `CALSAFE` is U-shaped with a broad, flat minimum, and one scalar recovers
most of the term. Qwen's best point leaves 0.0058 of ratio over the exact-scale ceiling,
against 0.045 with no factor, 87% recovered. SmolLM2's `CALSAFE`=2.0 leaves 0.016 against
0.129, 87.5%. The two models recover the same fraction while their terms differ 2.9x. The equal
fraction indicates that the factor acts on the mechanism rather than on one model's quirk.

The minimum is flat from 1.5 to 3.0 and its sides are shallow, so the factor does not need
tuning per model. On Qwen the ratio is 1.008 at 1.5, 1.009 at 2.0 and 1.008 at 3.0, and 4.5 is
0.0085 of ratio worse than the minimum. Below the minimum, the cost is clipping the tail. Above
it, the cost is resolution used on headroom no value reaches. The analytic bound takes that
failure to its limit, at 57.7x median overshoot.

Only Qwen's curve is mapped. SmolLM2 ran 1.25 and 2.0 alone, so 2 lying inside a flat minimum
there too is [expected]. On SmolLM2, 1.25 recovers 58% against 2.0's 87.5%, so software should
choose the factor high rather than low.

The residual after the factor is about what the random-column control costs: Qwen's 0.0058
sits just above `_calshuf`'s 0.0052. On Qwen the factor therefore removes nearly all of the
correlated clipping damage and leaves the term any perturbation of that size costs. A
per-column or per-layer estimator would have to reduce that residual, and neither is
implemented.

## The calibration window count, `NCAL`

`NCAL` is the number of disjoint calibration windows the colmax is frozen over. More windows
recover the same term as `CALSAFE`, more slowly, and the two parameters fall on one curve.

**Conditions:** `CALSAFE` held at 1.0.

| NCAL | Qwen ppl | x fp32 | ratio median | tightest | saturating | SmolLM2 ppl | x fp32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 13.4451 | 1.047 | 1.083x | 0.228x | 0.180% | 12.6872 | 1.133 |
| 4 | 13.1409 | 1.023 | 1.156x | 0.250x | 0.088% | - | - |
| 8 | 13.0340 | 1.015 | 1.224x | 0.281x | 0.043% | 11.7297 | 1.048 |

Four times the calibration tokens recovers 71% of Qwen's gap and 66% of SmolLM2's, against the
factor's 87%. The colmax has therefore not converged at two windows, and it is still rising at
eight. A larger calibration set is the expensive way to get what a multiplier gives.

Both parameters act through one quantity: the median ratio of the supplied colmax to the
call's own. Ordered by it, Qwen's points fall on one curve whichever parameter produced them.
The perplexity ratio falls to a minimum and rises again:

| median colmax ratio | x fp32 | setting |
|---:|---:|---|
| 1.083x | 1.047 | `NCAL`=2, `CALSAFE`=1.0 |
| 1.156x | 1.023 | `NCAL`=4 |
| 1.224x | 1.015 | `NCAL`=8 |
| 1.349x | 1.016 | `CALSAFE`=1.25 |
| 1.614x | 1.008 | `CALSAFE`=1.5 |
| 2.156x | 1.009 | `CALSAFE`=2.0 |
| 3.233x | 1.008 | `CALSAFE`=3.0 |
| 4.846x | 1.016 | `CALSAFE`=4.5 |

The points `NCAL`=8 (median 1.224x, 1.015) and `CALSAFE`=1.25 (median 1.349x, 1.016) are
indistinguishable in outcome. More windows give a better tail per unit of median looseness:
0.281x at median 1.224x, against 0.273x at 1.349x. The difference does not reach the model's
score. At equal looseness the two parameters are interchangeable, so software should
use the factor, which costs nothing.

**Not characterized:** any of these five cases:

- The two parameters combined.
- A per-layer or per-column factor. `CALSAFE` is one scalar for every column of every layer.
- A model outside the one activation regime measured. Both models are ~1.5B.
- A percentile over calibration rows, which moves the scale tighter, the wrong way for this
  tail.
- `NCAL` past eight windows, so where the colmax converges is not known.

[host-computed, arithmetic, two models, 2026-08-10]

## The frozen colmax from two device passes

Two device calls, the bootstrap, estimate each column's maximum from the entry's own int8
output. Every configuration above freezes the exact per-column maximum of the calibration
accumulator. No caller of `rocket_matmul_int8_rk3576_perc()` can see it, because the entry
writes int8 and the accumulator never leaves the NPU. Computing it on the host means a host int32
GEMM per calibration window per projection, which is the cost the offload exists to avoid.

The bootstrap runs two passes:

1. Pass one runs the entry at the analytic accumulator bound `127/(128*sum_k|B[n][k]|)`, which
   overshoots a real column by ~60x median and so cannot saturate. Each column's largest
   `|C8|`, read back, is a lower bound on that column's magnitude.
2. Pass two runs at that readback times a margin (1.5), so the int8 codes are used, and reads
   each column again.

A column that saturates in pass two keeps pass one's margined value. Both passes go through
the shift-0 integer ramp. They are dequantized by the scale requested rather than the gain
delivered, so the ramp's own error is inside the estimate.

The estimate is accurate. Its `|est/true - 1|` median over columns is 0.30% on Qwen2.5-1.5B
and 0.31% on SmolLM2-1.7B, flat across every shape. The worst single column is 2.4% / 8.3%,
with 0.14% / 0.37% of columns falling back from pass two. The table gives what that costs the
model.

**Conditions:** `CALSAFE`=2.0, both configurations in one process against one `fp32` run.

| model | `_cal` (exact frozen colmax) | `_calboot` (two device passes) | delta |
|---|---|---|---|
| Qwen2.5-1.5B | 12.9510, **1.0086x**, 95.41% top-1 | 12.9517, **1.0087x**, 96.02% | **0.000055** |
| SmolLM2-1.7B | 11.4165, **1.0199x**, 94.80% top-1 | 11.6192, **1.0380x**, 95.12% | **0.0181** |

A 0.3% estimator error is free on Qwen and costs 0.018 of ratio on SmolLM2, where it takes the
recovered fraction from 90.5% to 76.0%. The two models show the same split on three measures.
The control `_calshuf` costs SmolLM2 54% of its damage against Qwen's 12%. The scale-source
term itself is 2.9x larger on SmolLM2, and a sub-percent perturbation is 330x more expensive
there.

On this path a perturbation's cost follows its correlation with the columns that matter,
not its size. SmolLM2 is where that correlation is expensive.

`CALSAFE` is flat for an exactly frozen scale and not for an estimated one, and that sets the
default. Raising `CALSAFE` from 2.0 to 3.0 on SmolLM2 moves the exact configuration 0.0017 and
the bootstrap 0.016:

| SmolLM2, `CALSAFE` | `_cal` | `_calboot` | gap |
|---|---|---|---|
| 2.0 | 11.4165, 1.0199x | 11.6192, 1.0380x | **0.0181** |
| 3.0 | 11.4356, 1.0216x | 11.4591, **1.0237x** | **0.0021** |

The gap falls 88%. The bootstrap's recovery of SmolLM2's 0.129 term rises from 76.0% to
87.4%, the same fraction the exact freeze gets on both models. The estimator's 0.3% tightening
therefore falls on columns that were already near their clip, and a margin applied to all of
them absorbs it. The default of `ROCKET_RK3576_CALSAFE` is **3.0**, the top of the flat
minimum. Qwen's exact configuration is 1.008x at both 2.0 and 3.0, and its bootstrap gap at 2.0
is 0.000055. On Qwen, 3.0 being safe too is [expected].

Pass two is required, and the absolute error shows it. Pass one's signed median ratio is
1.499x / 1.523x against a 1.5 margin. Its `|est/true - 1|` is 49.9% / 52.3% median, and
563% / 230% at worst. At the analytic bound a column reads back a couple of codes. Pass one is
therefore unbiased in the median and far off per column
[host-computed, arithmetic, two models, 2026-08-11].

> [!NOTE]
> A signed median ratio that matches the margin does not show that a per-column estimator is
> accurate. Judge the estimator by its absolute error.

## The two bootstrap passes on the device

The bootstrap's median per-column error on the device is 0.28-0.48% at every `M`, against the
host simulation's 0.30% / 0.31%.

**Conditions:** H96 MAX M9, 2026-08-11. The operands are pseudorandom, with a deliberate ~30x
spread of `sum|w|` across the columns of one tile. No rotation is applied, because the
rotation's job is to make a real activation Gaussian and these operands already are. The
reference is the exact int32 accumulator on the host, which no frontend has.

| `M` | `K` | `N` | pass-1 `SHIFT` | pass-1 median &#124;err&#124; | est median | est worst | composed RMS rel | `worst_rel_err` | saturating |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1536 | 2048 | 39 | 53.1% | **0.455%** | 22268% | **15.3%** | 33.1% | 0.0000% |
| 512 | 1536 | 2048 | 39 | 6.9% | **0.283%** | 0.98% | **2.23%** | 0.417% | 0.0000% |
| 1 | 2048 | 2048 | 39 | 61.8% | **0.480%** | 611% | **18.1%** | 39.3% | 0.0000% |
| 256 | 2048 | 1024 | 39 | 12.2% | **0.296%** | 1.11% | **2.11%** | 0.655% | 0.0000% |

The device matches the host simulation, so the C ramp's integer gain, which the simulation
does not model, does not dominate the estimate's error. Pass one's signed median again falls on the margin
(0.97-1.63), while its absolute error is 6.9-61.8%.

The OUT_CVT shift is a six-bit field in DPU 0x40B4, and this path cannot drive the shift out of
it. Pass one's scale is ~6e-8, but the planner, `rocket_rk3576_plan_perchannel()`, divides the
scale it is handed by a `cmax` built on the same accumulator bound, `128*sum|w|+1`. With a NULL
bias the two cancel exactly, so `best_base` is the constant `127/INT32_MAX` whatever the
weights are. The device is therefore programmed with `SHIFT`=39 at every shape, 24 values
inside the field, and the per-column `C` comes out at its `cmax` ceiling. The shift stays 39
over `sum|w|` from 128 to 1.7e7. The frozen call programs 27-34.

Nothing in `librocketnpu` refuses a shift past the field. `rocket_rk3576_requant_params()` has
no bound check, and the RK3576 emitter writes `out_cvt_shift` unmasked, where the RK3588 path
masks it with `& 0x3F`.

On this path `M` has no shape constraint and a large accuracy one. A call at `M`=1 runs with no
row rule (the height<4 rule is the RK3588's), and its composed error is 6.9-8.6x the prefill
figure. The cause is the tile rather than the estimator. A column's frozen colmax is a maximum
over `M` accumulator rows: tight at a prefill `M`, and one Gaussian draw at `M`=1. The entry
plans one `(MUL, SHIFT)` per output-channel tile, so the spread of `scale_n` inside a tile sets
every column's `C` resolution.

The `ggml-rocket` frontend sends no call with `M` below `rocket_min_m()` (`ROCKET_MIN_M`,
default 32, never below 4). `M`=1 is therefore reachable only through the library entry, and
not through the frontend. **Not characterized:** `M` 32-255, which the frontend does reach.

At `M`=1, freezing from the exact column maximum is 13x worse than the bootstrap. The table
crosses the colmax source with the `N` tile:

| `M` | `K` | colmax source | `NT` 2048 | `NT` 256 | `NT` 32 |
|---|---|---|---|---|---|
| 1 | 1536 | bootstrap | 15.3% | 15.0% | **10.7%** |
| 1 | 1536 | exact | 201% | 148% | **70.7%** |
| 1 | 2048 | bootstrap | 18.1% | 18.1% | **14.5%** |
| 1 | 2048 | exact | 181% | 121% | **63.4%** |

The bootstrap's readback has a minimum of one code: a column that reads back zero is credited
one. That minimum bounds its colmax from below, and so bounds `scale_n` from above. The exact
maximum has no minimum. At `M`=1, a column whose single accumulator draw is near zero is handed
an enormous scale. The entry takes `best_base = max_j(cs_j/cmax_j)` over the whole tile, so one
such column costs every other column in that tile its `C`.
[HW sweep, H96 MAX M9, 2026-08-11]

The estimator's worst per-column error, the 22268% over-estimate, comes from the same one-code
minimum that keeps the shared ramp usable. The library's source gives the minimum's reason as
a too-tight scale costing a model. At low `M`, keeping the shared ramp usable is its larger
effect. A smaller tile
lowers the error monotonically and does not close it. Sorting the columns and tiling them
therefore reduces the `M`=1 composed error ~1.43x at 64 tiles on this path.
[HW sweep, H96 MAX M9, 2026-08-11]

## The `rk_hadamard_ok()` shape check

`ggml-rocket`'s `rk_hadamard_ok()` admits the two exact rotations, a power-of-two `K` and
`K = 60*2^k`, and nothing else. Qwen2.5-1.5B contracts every projection over `K`=1536, which is
`3*2^9`, not a power of two, and `1536 % 60 = 36`. On its own the check therefore refuses the
whole model whose 1.047x set the scale-source term's size. The RK3576 path admits a `K` through
`rk76_rot_ok()` instead. That check adds the block-diagonal rotation,
`rk_hadamard_rotate_blocks()`, for any `K` whose largest power-of-two divisor is at least 32.
Every Qwen2.5-1.5B projection takes that fallback (§"Accuracy of the W8A8 path inside a
model").

Wherever `K` was not a power of two, the accuracy simulation used the block-diagonal H over the
largest power of two that divides `K`. That H is still orthonormal, so the product reconstructs
exactly. It spreads an outlier over its block rather than over the whole contraction. It is
therefore the weaker construction, and a fallback rather than a replacement.

> [!NOTE]
> A frontend's shape check, written from where one construction is available, can disagree
> with the construction an accuracy figure was measured in. `rk_hadamard_ok()` alone disagrees on
> a whole model. Before applying an accuracy figure to a model, check which rotation the frontend
> applies at its `K`.

## Evidence

### The W8A8 path against the CPU at `pp512`

The two CPU denominators ran in the same boot, interleaved with the NPU arm, two processes each.
The table's per-process values are those two processes.

Without `assigned-clock-rates` in its device tree, this board runs the NPU at 786 MHz. The
2026-09-30 image, which assigns 594 MHz, postdates every measurement in this note
([../chips/rk3576.md](../chips/rk3576.md#npu-clock-rate)).

The offloaded count per micro-batch is 166 on the profiled runs. A `pp512 -r 1` run makes 332
calibrating calls over its two forwards. The timed `pp2048` pass at `-ub` 512 makes 664 converged
calls over four. The two weights missing from the 168 are the final block's gate and up. A
`llama-bench` calibrating forward at `-ub` 2048 logs 750080 columns. That is 28 blocks of 27648
columns, less the final block's FFN (8960 + 8960 + 6144) and with no output head
[host-computed].

### Flash attention past a 512-token prompt

The `ggml-rocket` source chain was read before the configuration ran.

### Time and wall-clock share of the offloaded W8A8 path

Each test point is one process, with the governor pinned to `performance` and `ondemand`
restored after. [HW sweep, H96 MAX M9, rocket 1.6.0, measured 2026-08-11]

The `M`=512 re-measurement ran three processes at `-ub 512`, 2988 library calls each, and reads
per call, in ms:

| bucket | 2026-08-11 | 2026-08-13 |
|---|---:|---:|
| `sumabs` | 3.62 | 3.629 |
| `packB` | 7.51 | 7.500 |
| `coeff` | 0.219 | 0.216 |
| `gen` | 0.044 | 0.0454 |
| `alloc` | 10.63 | 10.640 |
| `stamp` | 2.13 | 2.121 |
| `submit` | 0.148 | 0.147 |
| `wait` | 4.75 | 4.745 |
| `read` | 3.65 | 3.601 |
| `packA` | 1.87 | 1.771 |
| accounted | 34.57 | **34.42** |

Process-to-process spread within the 2026-08-13 run is 0.04-0.6% on every bucket except `packA`
(9.8%) and `read` (3.4%). The `pp2048` wall-clock rate reads 12.33, 12.37 and 12.35 t/s. That
spread is the resolution a comparison on these buckets has to clear. It does not license quoting
a level from another session, only the observation that these particular levels held.
[HW sweep, H96 MAX M9, 2026-08-13]

#### The M-independent intercept

The supplied sums and the resident weight cube remove `sumabs` and `packB` (§"Supplied sums,
the resident weight cube and the BO pool"). That removes the whole intercept, because the other
`M`-independent buckets hold nothing. The `alloc` bucket belongs to a different optimization.

### Supplied sums, the resident weight cube and the BO pool

#### Bit-identity

The supplied-sums and recomputing forms are bit-identical on the device. With
`ROCKET_RK3576_WSA` either way, `llama-perplexity --chunks 4` returns PPL 8.7433 +/- 0.76052 to
every printed digit, on the same binary and the same text. The supplied-against-computed
configuration of `tests/rk3576_mm_requant` measures 0 of 344064 elements differing, over six
shapes at two column-scale spreads.

The resident cube and the pool are bit-identical too. `--chunks 4` prints PPL 8.7433 to every
digit in both WDEV modes. The requant test's resident, pooled(x2) and forced-16-tile
configurations measure 0 differing bytes, and the test list passes 60/60.
[HW sweep, H96 MAX M9, test points `w{0,1}p{0,1}_p{1,2,3}`, 2026-08-19]

#### Per-call buckets and throughput

The three configurations were scored over 3x3 interleaved processes at `-ub 512` on one binary
(2988 calls). The interleaved `pp2048` throughput is from n=3 pairs resolving ~3.3%.
[HW sweep, H96 MAX M9, test points `w{0,1}p{0,1}_p{1,2,3}`, 2026-08-19]

#### The pool at a pinned layout

The 3x3 run gives the pool +0.9% of `pp2048` throughput, from n=3 pairs against a ~3.3%
instrument resolution. The pinned-layout figure, +1.86%, is about twice that. Collapsing the
per-process layout draw with `setarch -R` on both configurations brings the resolution to
0.358% (re-derived from the configuration's own test points), which makes the level directly
measurable. The run is four interleaved pairs in one invocation on one binary, with
`ROCKET_RK3576_BO_POOL` the only treatment. The per-pair gains are +2.115, +2.112, +1.385 and
+1.820, mean +1.8579%. [HW sweep, H96 MAX M9, 8 test points, 2026-08-20]

One of the four pooled test points ran 0.501% slow with its refill count at its configuration's
mean. The slow test point is not the page-color draw, and it is unexplained. It biases the
figure downward, since dropping it widens the gap to +2.028%.
[HW sweep, H96 MAX M9, 8 test points, 2026-08-20]

### Per-process variation in prefill throughput

#### Three measurement series on 2026-08-13

The `pp2048` value of the `sumabs` pass is not measurable from process-to-process throughput.
The first measurement series measured the knob that enables the pass as +6.70% of `pp2048`, and
the rebuild as -6.96%. Re-run with the order reversed (the treated configuration, the isolation
control, the control, one process each), the knob flip measured +1.7% and the binary flip +2.0%
the other way, while every bucket reproduced its attribution. Throughput moves by up to ~5%
between processes with every bucket flat and no reproducible structure. Fourteen processes over
three measurement series show it:

- The first series' five same-test-point sequences were flat inside their blocks (spread
  <= 0.4%, ctl 12.33/12.37/12.35, arm 12.28/12.23/12.27, iso 11.48/11.50), with monotone
  steps between blocks.
- The reversed series declined monotonically by position (12.24 -> 12.03 -> 11.79).
- A same-day A/A, the treated test point three times inside one invocation, measured 11.77,
  12.26 and 11.69. It broke both patterns while its buckets held (`quant+rot` within 3.4%,
  `packB` 1.1%, `wait` 0.2%, `sumabs` 0).

The same test point spans 11.69-12.28 within one day and +-4.5% across days. Repetitions inside
a process agree to +-0.25%, so the term is a per-process constant, fixed at process birth. It
lives in the unbucketed ~78% of prefill (attention, norms, rope, the CPU-placed matmuls). On
these series its mechanism was [unknown], and it defeated every ordering explanation tried on
it: test-point identity, invocation position, thermal soak and idle reset.
[HW sweep, H96 MAX M9, fourteen profiled processes over three measurement series, 2026-08-13]

A throughput comparison needs interleaved configurations and enough processes to beat a ~2%
per-process sd read from its own controls. An optimization's durable cost is its per-call
bucket. The buckets are stable to 2-3.5% across sessions, where the effects of interest are
10-43%.

#### Counter attribution

An A/A triple ran under `perf stat` on 2026-08-19 in one invocation, with the governor pinned
and an equality-gated epoch:

| process | throughput, t/s | inter-call `gap`, s | buckets, s |
|---|---:|---:|---:|
| 1 | 12.57 | 545.5 | 136.6 |
| 2 | 12.56 | 545.2 | 137.1 |
| 3 | 12.11 | 569.2 | 135.0 |

The slow process has its whole excess in `gap`, with the smallest bucket sum of the three. The
term is therefore between calls inside the span, not in the bench scaffolding outside it.
Cycles track the throughput, so CPU frequency is excluded, and the constant is DRAM traffic from
L2 misses. [HW sweep, H96 MAX M9, test points `aap_p{1,2,3}`]

#### Flat series and the draw rate

The term is not an unconditional per-process constant: ten consecutive fresh processes can come
back flat on every counter. A ten-test-point series ran on 2026-08-19 with one binary,
interleaved 5+5, the governor pinned and 4 threads on the A72 cluster. Across its five plain
processes, `l2d_cache_refill` max/min measures 1.0061, against the +23.7% the A/A triple showed
between two of three. Cycles measure 1.0013, instructions 1.0002 and `l1d_cache_refill` 1.0014,
and the throughput spans 0.15% (13.68-13.70 t/s).
[HW sweep, H96 MAX M9, test points `htlb_{n,a}_p{1..5}`]

Whatever draws the term did not draw on this series. A throughput comparison run under these
conditions therefore resolves far better than the ~2% per-process sd implies. The sd is a
property of the series, not a constant of the board. It must be read from each series' own
controls rather than inherited. [HW sweep, H96 MAX M9, test points `htlb_{n,a}_p{1..5}`]

The series held a 1024-page hugetlb pool and ran `compact_memory` before it. Compaction
defragments and shrinks the physical pool every process then draws from, and it is the leading
suspect for the flatness [hypothesis]. The protocol and epoch rivals to it are eliminated
below. [HW sweep, H96 MAX M9, test points `htlb_{n,a}_p{1..5}`]

The term is a draw, not a spread, and it is present on every epoch measured. The 30 archived
processes behind the body's 26-and-4 split come from seven series.
[HW sweep, H96 MAX M9, test points `nopool_p{1..5}`]

A five-test-point no-pool control on the current binary (2026-08-19, governor pinned,
`taskset -c 4-7`) measured 13.73, 13.72, 13.71, 13.71 and 13.26 t/s. The fifth process
reproduces the A/A triple's signature exactly: `l2d_cache_refill` +15.32% against the mean of
the other four, on instructions +0.05% and `l1d_cache_refill` +0.01%. Its elapsed time is
+3.11%, with every timed bucket flat to +-0.2%, so the term survives the binary epoch.
[HW sweep, H96 MAX M9, test points `nopool_p{1..5}`]

A CV or a span at n=3-5 measures whether a draw fell in the sample, not the term's size. The
CVs of these series therefore range 0.05% to 2.15% with no setting explaining them.
[HW sweep, H96 MAX M9, test points `nopool_p{1..5}`]

#### The convert-loop draw

The second per-process draw runs in the weight-convert loops, in a different address space from
the throughput draw. The empty band between the two `wt_rotate` modes is 7.5x the next-widest
gap in the sample. The work is byte-identical every run (166 convert calls, 2988 library calls,
6876 tiles, 27504 submits), so the draw is in timing, not in the workload.
[HW sweep, H96 MAX M9, re-read of 30 processes across seven series]

On the epochs where `wt_quant` is still a scalar loop, it is two-state with `wt_rotate` (5800 or
6558 ms, its own gap 739 ms against a next-widest of 8 ms). The two loops agree on the mode 12
of 12. One state the process is handed spans two independent loops, which places it in the
allocation-layout family rather than in code alignment. The draw does not move the `pp2048`
throughput (-0.02% and -0.09% within the two flat series), because it runs in weight
conversion, outside the timed reps. [HW sweep, H96 MAX M9, re-read of 30 processes across seven
series]

The draw is useful as a probe: it is binary per process at a ~50% base rate, with cleanly
separated modes. Every profiler log prints it, so it is readable retroactively at no cost. The
throughput draw, by contrast, is rare enough (4 of 30) that n=5 sees nothing 24-53% of the time.
A 7.3% `wt_rotate` difference between two n=3 samples was the mode mix, not the epoch.
[HW sweep, H96 MAX M9, re-read of 30 processes across seven series]

The convert-loop mode is a virtual layout draw, and `setarch -R` pins it. Running the same
binary with ASLR disabled collapses the draw to 0 low of 16 at a full spread of 0.134%. An
interleaved ASLR-on control measures 4 low of 16 at 11.756%. The variance collapses 88x, at
one-sided Fisher p=0.0506 on the mode mix alone. The state a process is handed is therefore
set by the layout ASLR randomizes, not by which physical frames it draws.
[HW sweep, H96 MAX M9, 32 interleaved test points on one binary, governor pinned, dial armed by
address readback]

The convert-loop mode cannot screen for a slow throughput draw. Within the same ASLR-on control,
mode-low test points are bench throughput outliers 2 of 4, against 4 of 12 for mode-high. At
one-sided Fisher 0.489, the two draws are independent [measured]. On a comparison that touches
the convert loops, `setarch -R` removes an 11.8% per-process nuisance draw but does not select
the fast state. The fixed level (11907-11923 ms) does not overlap the ASLR-on modal cluster
(11874-11895). [HW sweep, H96 MAX M9, 32 interleaved test points on one binary, governor pinned,
dial armed by address readback]

High is a band, not a state. The ASLR-on high draws separate into that modal cluster (9 of 12),
an 11930 and a pair near 12015. Low is a discrete state: 10752, 10753, 10753 and 10754 over four
independent processes, a spread of 0.019%. A mode reading costs one `pp512 -r 1` process,
99-103 s, against 630 s for a throughput test point. [HW sweep, H96 MAX M9, 32 interleaved test
points on one binary, governor pinned, dial armed by address readback]

#### The `pp512` bench under `setarch -R`

The bench's per-process throughput draw is virtual too. The term decomposes into a draw the
layout fixes plus a slower one that survives the pin. The same 32 test points each ran a
`pp512` bench. The series was two runs of the harness 47 s apart, so it is scored within an
invocation. Pooling across that boundary turns the level step below into apparent per-process
scatter, and inverts the answer. [HW sweep, H96 MAX M9, same 32 test points, no additional board
time]

Within an invocation, the ASLR-on control shows 6 of 16 throughput outliers (>= 1.0% off its own
median), against 0 of 16 with ASLR off, at one-sided Fisher p=0.0088. The ASLR-on outliers form
a discrete band at -2.4% to -4.1%, 3 of 8 in both blocks, with full spreads of 4.366% and
4.188%. The ASLR-off full spreads are 0.592% and 0.329%. The layout a process is handed
therefore sets its bench throughput as well as its convert-loop mode. The two are independent
draws in that one address space.
[HW sweep, H96 MAX M9, same 32 test points, no additional board time]

The magnitude and the empty-gap signature match the `pp2048` draw's (2.8-3.6% off median, 4 of
30) closely enough to suspect one phenomenon. This run is `pp512 -r 1`, though, so that
identification is [hypothesis]. One component does not collapse: under ASLR off the bench
throughput is flat inside an invocation, and its level moved +2.508% between the two
invocations. The interleaved ASLR-on configuration did not move (+0.031% overall, high band
-0.230%, low band +0.484%), so a system-wide drift is excluded. The convert loop's ASLR-off
level did not move either (-0.030%, 0.134% across all 16). [HW sweep, H96 MAX M9, same 32 test
points, no additional board time]

#### The `pp2048` throughput under `setarch -R`

The per-process throughput draw is virtual at `pp2048` as well, the depth the draw was fitted
on, so it is not page luck. The run is 16 interleaved `pp2048 -r 3` test points in one
invocation, with both configurations otherwise identical. The ASLR-on control reproduced the
draw: 3 of 8 outliers at a 4.966% full spread. Two of them sit inside the 2.8-3.6% band the
corpus recorded, and the third is at -4.487%. The ASLR-off configuration is 0 of 8 at a 0.292%
spread, a spread ratio of 16.99x against ~10x at `pp512`.
[HW sweep, H96 MAX M9, test points `walr_{on,off}_p{1..8}`]

The mechanism shows on the counter: `l2d_cache_refill` CV 9.699% on -> 0.099% off, a ~98x
collapse, against 17x for the throughput spread. The mode mix alone is underpowered at this n
by construction (Fisher 3/8 vs 0/8 = 0.1000), so the magnitude is what the result rests on.
"Full spread" is range/min throughout, so the 16.99x is like-for-like against the `pp512`
figures, and by the median the same test points give 16.25x. The two medians are 0.109% apart,
and the ASLR-on non-outlier band's median is 13.72 exactly.
[HW sweep, H96 MAX M9, test points `walr_{on,off}_p{1..8}`]

The configurations' means differ by 1.33% only because three slow draws drag the control's
mean. `setarch -R` therefore removes the slow tail and does not move the level. Its scope is the
per-process draw only, because it pins the virtual layout, not physical frames. It cannot
separate "the draw is virtual" from "frame assignment goes near-deterministic once the layout
is fixed". The +2.508% between-invocation step is untouched by it.

The sd of 0.087% is what makes sub-1% host optimizations measurable on this board.
[HW sweep, H96 MAX M9, test points `walr_{on,off}_p{1..8}`]

#### Physical frames

The physical frames a bench is handed are a fresh set every process. Reading
`/proc/<pid>/pagemap` concurrently with a live `pp512` bench, at fixed virtual addresses under
`setarch -R`, gives the (virtual address -> physical frame) map directly. Two test points
compare by positional agreement, the fraction of commonly sampled addresses whose PFN is
identical. That measure is a magnitude, and it is indifferent to physical contiguity. Over 8
test points in 4 invocations the comparison is total (one VMA layout, `common = union = 12655`
on every anon pair). [HW sweep, H96 MAX M9, test points `armj2_i{1..4}_c{1,2}`]

Anon agreement is 0.00000 on 26 of 28 pairs and <= 0.00063 on the other two, against a run-count
standard error of 0.0094. It is 0.00000 within an invocation, where the throughput is flat, as
well as between invocations. No timescale therefore exists on which the anon frames repeat.

File agreement is 1.00000 on all 28 pairs. The measurement answers frame identity, not whether
physical memory matters. A statistical property of the frame set can still track the throughput
while every identity differs. [HW sweep, H96 MAX M9, test points `armj2_i{1..4}_c{1,2}`]

Anon allocation is contiguous in buddy blocks, on a kernel with no THP. The mean run is 69.6
pages (278 KiB), the median 32 pages, and 96.8% of run lengths are powers of two over orders
4-9. The sample is every 16th page, so 39.7% of runs sit at that sampling limit and shorter ones
are unresolved. Contiguity is why a 16-bin L2 page-color histogram has no information on this
workload. Inside a run the color walk is deterministic, and long runs cover the 16 colors
near-uniformly, so the histogram is flat by construction.
[HW sweep, H96 MAX M9, test points `armj2_i{1..4}_c{1,2}` and `armjprice`]

A stride-1 read measured chi2 0.12 on 15 df at n=202,459, sub-multinomial by 10.9x. A color
histogram needs fragmented allocation to be informative, and allocation here is not fragmented.
A sampling stride that is a multiple of 16 makes it worse. That stride advances the color by
zero inside a run and produces a large chi2 that is pure aliasing.
[HW sweep, H96 MAX M9, test points `armj2_i{1..4}_c{1,2}` and `armjprice`]

#### The between-invocation component

The between-invocation step is not a property of starting a new invocation. Across four
invocations of two ASLR-off `pp512` test points (60 s apart on an otherwise idle board, one
binary, `performance`), the boundaries move -0.082, +0.123 and +0.082%.

Decomposed, the within-invocation sd is 0.1085% (4 df), and the sd of the four invocation
means is 0.0854% against 0.0767% expected from within-noise alone. The between component is 0.0374%,
F = 1.238 on 3,4 df, with a 95% upper bound of 0.237%. The bound is an order of magnitude under
the +2.508% step. That step's two invocations were separated by a long stretch of other work.
[HW sweep, H96 MAX M9, test points `armj2_i{1..4}_c{1,2}`]

The bound holds only under a continuous-nuisance model. The throughput term on the RK3576
behaves as a discrete draw, with 37.5% outlier rates at the process level. Against a draw, 0
fires in 3 boundaries bounds the per-boundary rate only at 0.63, which excludes nothing. At a
rate of 0.375 it takes seven boundaries to say anything. An sd decomposition is not a bound on
a draw, and the flatness of these four invocations describes them without bounding the step.
[HW sweep, H96 MAX M9, test points `armj2_i{1..4}_c{1,2}`]

#### 2 MB pages

The hugetlb configuration reaches 601-606 2 MB pages (~1.2 GB of anon) at the bench's peak,
against 0 without it. Its 0.64% loss has a consistent sign in 5 of 5 pairs, at ~4x the 0.15%
within-configuration spread. [HW sweep, H96 MAX M9, test points `htlb_{n,a}_p{1..5}`]

Elapsed time is +0.53%, cycles +0.20%, and `l2d_cache_refill` is unchanged at -0.2%. The host
buckets move with the throughput (mm_wall +1.8%, entry +1.5%, alloc +1.4%, packB +1.3%). The
`read` bucket at -9.4% and `wt_rotate` at -4.1% sit inside their own 7.3% and 12% process
spreads, and show nothing. The series had no slow process, so it measures the setting at the
flat operating point. [HW sweep, H96 MAX M9, test points `htlb_{n,a}_p{1..5}`]

#### Excluded causes

- *Physical page placement.* The physical-page mechanism says the L2 is physically indexed.
  The physical pages a process is handed at birth would then set its conflict rate for life.
  Over 16 interleaved test points, `setarch -R` collapses the `l2d_cache_refill` CV ~98x and
  the throughput spread 17x. The draw is therefore virtual, not page luck.
  [HW sweep, H96 MAX M9, test points `walr_{on,off}_p{1..8}`]
- *The L2 geometry the page mechanism assumed.* The 512 KiB, 512-set L2 the mechanism was
  argued from is the A53 cluster's. Every test point here runs `taskset 4-7` on the A72, whose
  L2 is 1024 KiB, 16-way and 1024 sets: 16 page colors, not 8.
  [HW sweep, H96 MAX M9, test points `aap_p{1,2,3}`]
- *One shared locality term.* The same A/A triple shows the q8/L1d term moving the other way in
  the slow process. The two locality terms are independent draws, and both are virtual.
  [HW sweep, H96 MAX M9, test points `aap_p{1,2,3}`]
- *CPU frequency.* Cycles track the throughput in the slow process (§"Counter attribution").
- *A system-wide drift, for the `pp512` level step.* The interleaved ASLR-on configuration did
  not move (§"The `pp512` bench under `setarch -R`").

### L1d set congruence across rebuilds

#### The 2026-08-19 attribution

The frontend commit that handed the entry its weight sums moved `quant+rot` 32.6-32.8 ->
45.6-47.2 s a process, locked to the test point across days. That diff's only code is in the
weight-convert path. The counters plus the disassembly separate L1d congruence from codegen in
one pair. [HW sweep, objdump, H96 MAX M9, 2026-08-19]

Phase stamps applied to both frontends put the whole move in `q8`: 96% of the delta, while copy,
rot and max held within 3.1%. The stamps are equality-gated: the old binary with and without
stamps prints PPL 8.7433 to every digit. [HW sweep, objdump, H96 MAX M9, 2026-08-19]

The set-level explanation is a hypothesis, and it fits every observation. It explains why
page-padding one allocation moved the level partially (42.4 s) and could not restore it.
[HW sweep, objdump, H96 MAX M9, 2026-08-19]

A third build clears the commit itself. The same diff rebuilt against the newer library (a
different link state) measures `q8` at the fast level in all three of that epoch's test points.
The diff was one way to change the congruence. Five builds are on record:

- Old frontend, old library: fast.
- New frontend, old library: slow.
- Old frontend, new library: fast.
- New frontend, new library: fast.
- The resident-cube frontend as committed, built as the `libggml-rocket.so` whose md5 begins
  `427aee98`: fast, with `quant+rot` at 31-33 s across its nine test points.

[HW sweep, objdump, H96 MAX M9, 2026-08-19]

After the NEON quantize, the `--chunks 4` equality test point is per chunk identical to the
prior epoch. The other frontend buckets stay within +-1%, and the throughput measures +1.9%
(recorded, not a between-process measurement). [HW sweep, H96 MAX M9, 2026-08-19]

#### The 2026-08-13 investigation

The reversed series shows the buckets locked to the test point and independent of position.
With the knob on, `sumabs` is 10.83-10.85 s a process, and 0 with it off, at every position on
both days. The knob moves `accounted`, `entry` and `calibrate` by its own size and nothing else.
[HW sweep, H96 MAX M9, fourteen profiled processes over three measurement series, 2026-08-13]

The `quant+rot` binary regression is real. The same diff moved that bucket 32.6-32.8 s ->
45.6-47.2 s a process (+43%), seven new-binary readings against three old-binary ones across
two days and three series. It sits on a function the diff does not touch: `rk76_quant_act`
reads the activation and no weight. The regression therefore belongs to the binary. The
candidates were codegen, or the allocation layout moved by one added per-chunk vector, on a
core whose host loops are L1d-set sensitive. [HW sweep, H96 MAX M9, fourteen profiled processes
over three measurement series, 2026-08-13]

The interior is mapped by a diagnostic patch that stamps each phase, its own binary epoch,
equality-gated at `--chunks 4` PPL 8.7433. Per process at ubatch 512, `q8` (the quantize loop)
is 48%, `rot` (Hadamard) 36%, `max` 12% and `copy` 4%. The four sum to `quant` within 0.14%,
which is the stamps' own overhead, self-reported as the difference. The discriminating tests
are the same stamps in the old binary, differenced per phase, and the `perf stat` cache-counter
pair. [HW sweep, one test point each, 2026-08-13]

The same diff also doubled `wt_quant`, the once-per-weight cache-insert quantization on the
convert line. It went 5.8-6.6 s -> 10.9-12.5 s a process across all fourteen logs. That cost is
outside the timed reps (first-touch), so it costs process startup rather than `pp2048`. The
convert line printed it in every log of the first series.
[HW sweep, H96 MAX M9, fourteen profiled processes over three measurement series, 2026-08-13]

#### Excluded causes

- *The moved allocation's alignment.* A diagnostic patch page-pads the diff's one new `sumabs`
  vector, the sole change against the default source. That configuration measures `quant+rot`
  at 42.4 s, 29% above the old level, so the regression is not that allocation's alignment. The
  7-10% it sits below the new-binary band is 2-3x the bucket reproduction spread, so a minority
  layout slice is plausible and open. [HW sweep, one test point each, 2026-08-13]
- *Codegen.* The two binaries' `q8` loops are instruction-identical (§"L1d set congruence
  across rebuilds" in the body).

### Share of prefill wall-clock time

Before the `perf` attribution, each candidate for the 78% was a [hypothesis]:

- CPU attention, with FA declined
- The norms
- Rope
- The scheduler
- Every CPU-placed matmul

The per-process term the gap accumulator exposes still had no mechanism at that point.
[HW sweep, one test point, 2026-08-13]

A two-point throughput comparison of ubatch 512 against 1024 measured 0.2%, which resolves
nothing against the configuration's +-15% spread.

The entry's share is 664 x 36.40 ms = 24.17 s of the 169.7 s prefill, 14.2%. The intercept's is
664 x 11.4 ms = 7.57 s, 4.5%, which is also 31.3% of 14.2% [host-computed].

### Prefill time by function

The attribution run took 803.0 s and 509,684 samples. It was recorded with `perf record` and
grouped by event group [HW sweep, Qwen2.5-1.5B F16 on llama.cpp b10356, `-p 2048 -ub 512 -r 3`,
`ROCKET_INT8=1`, `ROCKET_FLASH_ATTN=0`, `taskset -c 4-7 -t 4`, governor pinned].

At `pp512`, `pp512 -r 1` reports 332 of 332 calls as calibration forwards, confirmed directly.

### The W8A8 path against ggml Q8_0 at `pp2048`

The per-pair ratios are 1.30 / 1.34 / 1.30. The 1.31x is neither the tie the `pp512` figure gave
nor the 2x of the F16 control. [HW sweep, H96 MAX M9, rocket 1.6.0, llama.cpp b10558,
`ggml-rocket` build of 2026-08-19, measured 2026-08-23]

#### Estimates before the host K-split measurement

A further optimization inside the entry is bounded by a path that already wins 1.31x. The
largest of them, a BO pool for the per-tile allocations, was capped at 4.2% [expected].

These estimates are at `-ub` 512 (`M`=512). The measured 2.26-2.37x of §"The host K-split:
prefill speed and perplexity" is at `-ub` 2048, so the two do not compare.

Moving `ffn_down` onto the NPU at the offloaded path's own per-unit rate was estimated to take
the wall-clock time from 50.2 s a forward to 22.7 s. The estimate is 2.21x on prefill, and the
port to about 4.3x against pure CPU [expected]. It counts only what the change removes. With
what the change adds enumerated, the estimate is 1.05-1.27x (§"The cost of a host K-split of
`ffn_down`" below). The figure sizes the term left on the CPU correctly, and omits the whole cost
of putting that term anywhere else.

#### The probe refusal

The one device probe of a chunk shape, `rk3576_matmul_probe ncap` at `M`=58 `K`=4480 `N`=1536,
refuses. Here `mm_int8_ex` builds a `conv_params_t` and drives the raw conv path with a
hand-picked plane. What refuses is the conv row-window planner, not the matmul entry. The
refusal is not a refutation of the chunked path.

### The cost of a host K-split of `ffn_down`

#### Entry cost and composition

The cost table's twelve test points cover `K` on a 64 step at `M`=512 `N`=1536. The first rep
is discarded, and the governor is pinned to `performance`. The composition error is one seed.
[HW sweep, H96 MAX M9, rocket 1.6.0, 2026-08-11]

The preview and the profiler reconcile on both counters. A reading against the wrong counter only
appears not to reconcile.

#### The estimate with the added terms enumerated

On 2026-08-11 the estimate put the host K-split at 1.05-1.27x on prefill, taking the port to
2.06-2.49x. Per `down` call at `M`=512, the entry costs 96 ms [measured]. The frontend's
rotate-and-quantize of a `K`=8960 activation costs 96-99 ms ([expected], 21.6 ns an element
measured at `K`=1536, scaled by `K` and by the 8/9 butterfly ratio of a 256-block rotation). The
dequantize plus f32 sum costs ~3 ms [expected]. The change adds 185-200 ms against the
221-280 ms the CPU GEMM costs. At 112 such calls a forward it saves 2.4-10.6 s of 50.2, which is
1.05-1.27x.

The estimate put the K-split at 2-6x the best in-entry optimization (4.2%), not 10-50x it, in
the same class. The largest single added term was the entry's own O(`N*K`) work, which a weight
cache would remove. On 2026-08-11 the entry had no weight cache, so `sumabs`, the cube `memset`
and `packB` were three passes over `N*K` on every call. Those passes were ~27 ms of the 96 at
`Ktot`=8960. Since 2026-08-19 the supplied sums and the resident weight cube take `sumabs` and
`packB` off a converged call (§"Supplied sums, the resident weight cube and the BO pool").

### The host K-split: prefill speed and perplexity

#### Prefill throughput

The configurations ran in two interleaved process pairs. Reading the three reps back out of
mean and stddev under the one-slow-two-fast structure gives a converged 32.3 against 12.3 t/s,
or 2.64x. The 2.64x is an inference from two summary statistics, not a per-rep reading. The
2.26-2.37x means are what is measured.

The pure-CPU F16 configuration behind the port's 1.96x costs ~26 minutes a test point.

#### Per-call terms

There are 0 redos over 39200 submits. The estimate before the measurement put the three terms
at `M`=512 at 96 +- 8 ms (`entry`), 96-99 ms (`quant+rot`) and ~197 ms (`calibrate`). Both
compute terms are below their estimated bands, in the direction that makes the optimization
larger. The cost estimate contained nothing about `calibrate`.

#### Perplexity

`nosplit` reproduces the separately recorded 1.011x of the W8A8 path, which is the harness
control.

The `machinery` configuration holds the rotation fixed by construction. At `K`=1536 the
block-diagonal fallback is `H_512` over three aligned blocks, which is bit-identical to what
three 512-chunks each take. The only things that move are the slicing, the per-chunk activation
scale and the summed partials.

#### Excluded causes

- *The frozen output scale, under-estimating.* The entry's saturation report does not fire on
  either split configuration. The rocket channel is confirmed reaching stderr in the same logs,
  so the per-column scale is not being outrun. The report's threshold is 0.1% of the output,
  not zero, so "does not fire" means "under a tenth of a percent".
- *The frozen output scale, wasting resolution.* Cutting `CALSAFE` 3.0 -> 1.2, so that a partial
  uses 106 int8 codes instead of 42, makes the model worse (15.84 -> 17.99). It also starts the
  report firing at 0.105-0.225% on four weights. The 3x headroom is not wasted resolution.

### Chunk depth and the rotation's block width

#### Chunk count and rotation width

The reading "more chunks is worse, monotone and steep" does not hold as stated: it is a confound.
The `k2048` configuration cuts `ffn_down` into six chunks and scores better than the default four.
The `k1792` configuration cuts it into five and scores decisively worse. The `chunk1024` test
point varies more than the count. Cutting a `K` like 8960 into more pieces at non-power-of-two
depths also narrows every piece's rotation, so the two move together. Held at a fixed `b`=256, the
count term is real and much smaller: 5 chunks score 21.81 against 24.35 for 7 chunks.

The rotation width does not order the set either. The `k4096` configuration has the widest
rotation reachable, 91% of `K` under a full `H_4096`. Every chunk is a power of two, and the chunk
count matches the default cut's. It scores 22.01, 8 of 8 windows worse and level with the
narrowest configuration measured. The `k6144` configuration has the fewest chunks and twice the
default cut's weighted width, and it is 2.11 PPL worse. Any model built on width and count alone
fails on both rows.

Both directions off 2048 are worse, so a search that starts from either end of the depth axis
alone finds a monotone trend.

#### Paired per-window comparison

The configurations are deterministic. Both controls reproduce across runs and boots to four
decimals. The reported `+-0.6` to `+-1.04` is therefore the spread of one configuration's own
windows, not run-to-run noise, and the sample is the eight windows. A paired-window reader backs
the per-window mean NLL out of the running estimate to compare the configurations pairwise:

| Configuration | Paired difference, mean NLL | Windows worse |
|---|---|---:|
| `k1792` | +0.3197 +- 0.0813 | 8 of 8 |
| `k1280` | +0.4300 +- 0.1057 | 8 of 8 |
| `k4096` | +0.3291 +- 0.0804 | 8 of 8 |
| `k2048` | -0.0925 +- 0.0566 | 3 of 8 |
| `k6144` | +0.1251 +- 0.0604 | 6 of 8 |
| `CALSAFE`=1.2 | +0.1270 +- 0.0793 | 5 of 8 |

Only the two narrow configurations, `k1792` and `k1280`, separate cleanly. The +-1.4 PPL moves
(`k2048`, `k6144` and `CALSAFE`=1.2) are 1.6 to 2.1 standard errors and are not established at
eight windows. A headline PPL that moves 1.4 at eight windows can rest on three of them.

### Per-row activation scale

The per-row configuration is worse than the default path on 0 of 8 windows. Its paired difference
is -0.4787 +- 0.1045 of mean NLL. [HW sweep, H96 MAX M9, rocket 1.6.0, 7.1.7, 2026-08-11]

#### The dequantize's fp32 association

An avoidable fp32 re-association in the dequantize moves the default configuration 15.8395 ->
15.9894. Folding `a_scale` into the per-column factor once, as the default path does, is not the
same float as multiplying it per element. At 0.4 of a paired standard error that is not an
accuracy change. It does cost the default path its bit-for-bit reproducibility across builds,
which is the property every configuration on this path is compared against. The per-tensor case
therefore keeps the original association, and the per-row case is a separate branch.

### Chunk depth under a per-row scale

None of the three per-row test points is worse than the `nosplit` control on more than 6 of 8
windows. The same three chunkings with a per-tensor scale are +0.4703, +0.3778 and +0.9003, at 8
of 8. Pairwise against each other, the three per-row configurations are within 1.1 se (4 of 8
worse each way). They behave as one test point, not a ladder.

Two candidates were tested for the term the chunk depth trades: the per-tensor scale's cross-row
reach, and the number of independently requantized partials. The first accounts for essentially
all of it. The second is bounded at <= 0.04 PPL over 4 to 7 chunks, so it is not a term. [HW
sweep, H96 MAX M9, rocket 1.6.0, 7.1.7, 2026-08-11]

### SmolLM2-1.7B

SmolLM2-1.7B is the other ~1.5B model in the set measured, and it disagrees with Qwen2.5-1.5B on a
term's size by 9x and by 38x. It is the second model for that reason.

### Llama-3.2-1B-Instruct

#### The model as a discriminator

The two earlier models differ in both their pretraining and their contraction geometry, so that
pair cannot say which one the 4.2x comes from. Llama-3.2-1B-Instruct has SmolLM2-1.7B's `K` set
exactly: `n_embd` 2048 and `n_ff` 8192, read off the GGUF headers, against Qwen2.5-1.5B's
1536/8960. It comes from a third pretraining family, with 16 blocks against 24 and GQA against
MHA. Holding the geometry, and therefore the K-split's own cut, fixed against one recorded model
makes it a discriminator rather than a third data point.

Four test points run at `--chunks 64` on `wiki-test-full.txt` with `-ub 512`, one process each [HW
sweep, H96 MAX M9, rocket 1.6.0, 7.1.7, `.so` md5 `ea40bc1b`, 2026-08-20]. That `.so` is the
`libggml-rocket.so` built from `ggml-rocket` 1210ea8.

Depth-matched at the 8 windows the other two ratios are measured at, this model measures 2.0100x.
The geometry account needs only the direction of the difference, and the direction holds at
matched depth. The whole-offload cost's variation over 8, 29 and 64 windows is one more sign that
eight windows are too few on this path.

#### `NCAL`=1 across the three models

At each model's own depth, `NCAL`=1 is indistinguishable from zero on two models, a matter of
statistical power. Over windows 3-64 this model puts `ncal1` at +0.0074 +/- 0.0025 se above `ctl`,
3.0 se from zero. It is worse on 41 of 62 windows, sign test two-sided p = 0.0151. Qwen2.5-1.5B
separates the same term at 0.56 se and SmolLM2-1.7B at 0.55 se, which is what the reading "does
not reach either model" reports. Depth-matched, the three agree: 0.69 se from Qwen2.5-1.5B at
`d`=8 and 1.45 se from SmolLM2-1.7B at `d`=29.

Pooled by inverse variance at each model's deepest test point, the three are homogeneous (Cochran's Q =
1.360 on 2 df).

### Perplexity measurement resolution

#### The text

A re-run on a rebuilt `.so` reproduces its predecessor's cumulative sequence byte-for-byte.

The text, `wiki-test-full.txt`, is `grep -v "^ *$"` over wikitext-2-raw's `wiki.test.raw`. It is
1287656 bytes and 2891 lines, md5 `3ce70e93`, and holds 298938 tokens under Qwen2.5's tokenizer ->
583 windows at `n_ctx`=512. The file `wiki-test-small.txt` (md5 `e454e460`) is exactly its first
60190 bytes, which caps it at 29 windows.

Because the small file is a byte-prefix, the deeper text is a strict extension. Window *i* covers
tokens `[i*512, (i+1)*512)` taken from the start. The seam at byte 60190 falls at token ~15000,
against window 29 ending at 14848. The `--chunks` flag therefore sets the depth, with every
recorded test point preserved as a prefix. A `--chunks 64` control test point on the full text
reproduces all 29 recorded cumulative values of the 29-window control byte-for-byte, ending
[29]**8.5745**.

#### The spread

The paired per-window sd is 0.01853 over the first 29 windows and 0.01655 over the 35 beyond them.
These are homogeneous, so no phase correction applies past the `NCAL` calibration windows. At the
2-se resolution, additivity's 0.0057 needs 38 windows and the regime-transfer
difference-in-differences' 0.0051 needs 47, both reachable. [HW sweep, H96 MAX M9, rocket 1.6.0,
7.1.7, 2026-08-13] The 0.0057 is the setting's 29-window cost. At its 64-window cost, 0.00415,
additivity needs 70 windows at the 64-window sd of 0.01734 [host-computed].

### `AROW` without the K-split

Three configurations run on Qwen2.5-1.5B in one build and one boot, paired window by window, at
both depths. No configuration recorded before these separates the setting from the split. [HW
sweep, H96 MAX M9, rocket 1.6.0, 7.1.7, 2026-08-13]

The site map comes from the `[rk3576-cache]` line the driver already prints, with `ROCKET_DEBUG=1`
plus `llama-perplexity -v`. The line goes through `GGML_LOG_DEBUG` and is written to **stdout**.

#### Revision history

The ratio between the setting's and the split's terms is not a measurement, because its
denominator is not distinguishable from zero. It moved with the depth:

| Depth | Setting with the split off | The split's own cost | Ratio |
|---|---:|---:|---:|
| 29 windows | 0.571% | 0.026% | 22x |
| 64 windows | 0.415% | 0.151% | 2.7x |

#### Build term

The build term on this path is exactly zero, measured three times. The `nosplit_a0` test point
reproduces the recorded 8-window `nosplit` test point byte-for-byte on all eight of its cumulative
values. On the deeper text, both `ctl_L` and `nosplit_a0_L` reproduce their recorded 29-window
predecessors byte-for-byte on all 29. The 0.28% separating the two recorded `arow` levels is
therefore confined to code that only the `AROW` and split path executes.

### Per-row scale on the split weights only

The chunk loop already quantizes each chunk's slab separately with its own scale, so the mode is
one boolean at three call sites.

A 64-window level is not comparable to a 29-window one. Only the paired difference is. The deeper
test points split as windows 1-29 at -0.00064 +/- 0.00344 (reproducing the shallow pair to the
digit) and 30-64 at -0.00180 +/- 0.00280. The effect is therefore homogeneous across the text, and
the deeper windows are not its source.

The 2-se interval excludes -0.0057, the additive prediction from the setting's 29-window cost
with the split off, at 2.04 se. On 2026-08-13 that was read as refuting additivity. Compared at the
same depth, the prediction from the 64-window cost, -0.00415, sits 1.33 se from the point
estimate. At 64 windows neither additivity nor the null is therefore refuted [host-computed].
No depth reachable on this corpus can separate 0.127% from 0 (746 windows needed, 583
available). [HW sweep, H96 MAX M9, rocket 1.6.0, 7.1.7, 2026-08-13]

At 29 windows the setting's cost with the split off, 0.571%, is 8.9x the scoped mode's recovery,
0.064% at 0.19 se. At 64 windows the two are 0.415% and 0.127%, 3.3x [host-computed]. Neither
ratio is a measurement, because its denominator is not distinguishable from zero.

The upper end of the recovery is -0.00127 - 2 x 0.00217 = -0.00561 nats, and exp(0.00561) - 1 is
0.56% of perplexity [host-computed]. The 0.43% recorded for this bound on 2026-08-13 is the 2-se
resolution at 64 windows (§"Perplexity measurement resolution"), the interval's half-width
rather than its end.

### `AROW` cost in the per-call buckets

The run is six interleaved processes, `split` / `arow` / `split` / `arow` / `split` / `arow`,
after a discarded warm-up process. All 7 of 7 processes return rc=0, with 0 redos, a backstop
delta of 0 and zero `WARNING`s. [HW sweep, H96 MAX M9, rocket 1.6.0, 7.1.7, 2026-08-11]

#### Excluded causes

- *The per-tensor branch's second flat pass over the rotated activation.* It was the obvious
  mechanism for `quant+rot`'s cost, and it costs nothing measurable. The 2 GB a forward of
  streaming re-read is invisible next to the rest of `quant+rot`'s time.

#### Process-to-process variance

On the terms `AROW` touches, the buckets are a 5x tighter instrument than wall-clock time. Over
the three identical `split` processes, the host arithmetic buckets spread 0.35-0.82%, against the
wall-clock time's 4.1% on the same processes. The `entry` bucket spreads 9.38% (12.52% on the
`arow` configuration).

The matmul profile decomposes that spread over all six processes, at an identical 2940 tiles and
47040 submits in each:

| Matmul profile term | Range over the six processes |
|---|---|
| `packA` | 13295-24188 ms (+82%) |
| `read` | 19338-24581 ms (+27%) |
| `wait` | 35626-35883 ms (0.7%) |
| `submit`, `packB`, `sumabs`, `coeff`, `gen`, `alloc`, `stamp` | Inside 1.5% |

The variance is the host's activation pack and output readback, not the device. Neither term
follows the configuration, since both configurations span the same `packA` range. The last two of
the six processes have the two highest values, so the variance drifts within one run. The
interleave protects the measurement here: a blocked three-then-three design would have taken up to
50% of `packA`'s range as signal. [HW sweep, H96 MAX M9, rocket 1.6.0, 7.1.7, 2026-08-11]

### Resolution of `pp2048` wall-clock time

The run is five processes. The governor is pinned to `performance` under `taskset -c 4-7`,
thermals sit at 47-51 C, and the backstop delta is 0. [HW sweep, H96 MAX M9, rocket 1.6.0, 7.1.7,
2026-08-11]

| Process | Configuration | Converged s/forward | t/s |
|---:|---|---:|---:|
| 1 | `split` (discarded warm-up process) | 61.87 | 33.10 |
| 2 | `split` | 84.32 | 24.29 |
| 3 | `split` + `AROW` | 76.34 | 26.97 |
| 4 | `split` | 74.31 | 27.56 |
| 5 | `split` + `AROW` | 76.64 | 26.72 |

Reps 2-4 within each process agree to under 1%, for example 61.88/61.87/62.53 and
84.56/84.32/84.85. Rep 1 is always ~2x the rest, which is `NCAL`=2 falling on the warm-up plus the
first timed rep, as recorded. Between processes, the identical `split` configuration measures
33.10, 24.29 and 27.56 t/s, a 36% spread.

The two adjacent pairs disagree in sign: +11.0% for `AROW` on one, -3.1% on the other. Nothing
drifts inside a process, so the state is set at process start. What separates the 36% run from the
4.1% one is its timing. It ran within the hour in which a 3.4 GB second model was written to the
board's 7.9 GB of RAM. The page-cache candidate is otherwise untested.

Six further processes on the same board, build and pinning measure converged `split` at
32.75/32.80, 32.31/32.54 and 31.51/31.46 t/s, a 4.1% spread. The `arow` configuration measures
30.64/30.60, 31.81/31.86 and 32.47/32.46. The two configurations differ by -1.8%, the same sign
the buckets give and 3.7x the magnitude. One 36% observation is not a rate.

### Calibration cost in the W8A8 frontend

The arithmetic over the matmul profile's own call count agrees with the two device passes: 1644
device passes = 548 converged + 1096 bootstrap. The `accounted` total's 33.3% converged share is
45.0 s, against the frontend's 47.5 s `entry`, a 5.5% per-call overhead gap.

The table's second row is 119.71 ms × 772 calls = 92.4 s a process. With the scan's 78.9 s, it
gives the bucket's 171.3 s [host-computed]. The record of 2026-08-11 gave that
cell as 119.7 s, the per-call figure in seconds.

The device-and-loops half holds to 0.96% across processes, while the scan half spreads 10.37%.
After the loop swap (§"The calibration scan's loop form"), the spread of `calibrate` itself falls
5.20% -> 0.90%: removing the term removes the variance with it. [HW sweep, 2026-08-11]

### The calibration scan's loop form

#### Column-outer against row-major

Three interleaved processes a configuration, the governor pinned. All 7 of 7 processes return
rc=0, and the run has 0 redos and a backstop delta of 0. The `llama-perplexity --chunks 4` run
returns PPL 12.8615 +/- 1.17731 to every printed digit both ways. The check is needed because a
wrong column maximum on this path produces output of the correct size with plausible values. [HW
sweep, 2026-08-11]

The per-rep `samples_ns` of `llama-bench` separates the calibrating and converged regimes by
construction.

#### Byte against `int` accumulator

Three interleaved processes a configuration, the governor pinned. All 7 of 7 processes return
rc=0, and the run has 0 redos and a backstop delta of 0. On the calibrating rep all nine pairings
are negative, with magnitudes spanning -1.80 to -8.23 against a 4.0% control spread.

The -0.58% in `dequant` is unexplained. No mechanism in this change reaches it, it is 0.05 ms a
call, and it is visible only because `dequant` resolves 0.16%. [HW sweep, 2026-08-11]

#### Bucket levels across runs

The `calscan` bucket resolves 0.05% across three interleaved processes in one run. Its level moved
15.8% between two runs (10.72 -> 12.41) on a branch in which not one statement changed. That
branch was recompiled inside a function that had gained a second branch. The `calibrate` (128.73
-> 133.36) and `entry` (58.39 -> 60.38) buckets moved with it. A tight within-run spread is a
property of the binary, not of the machine.

#### Excluded causes

- *Accumulator cache residency, for the byte form's gain.* A residency reading needs the gain to
  grow with `N` (35 KB at `N`=8960 is past the RK3576's 32 KiB L1d, 1 KB at 256 is not). The gain
  is flat at every `N` and largest at 256.
- *The vector max alone, with nothing from the narrowing.* That reading needs 1x, and the same
  three test points refute it.
- *TLB reach, for the column-outer scan.* At `M`=2048, a column walk over `N`=1536 spans 3.1 MB =
  768 pages, which fits the RK3576's L2 TLB. At `N`=8960 the walk spans 18.3 MB = 4480 pages,
  which does not. A TLB reading predicts that `N`=1536 is the cheaper, and it is the more
  expensive [HW sweep, H96 MAX M9, 2026-08-11].

At `N`=256, set count is confounded with a span that fits L2. The 1536-against-8960 pair has no
such confound, and it is the test point that decides between the set-conflict and TLB readings.
[HW sweep, H96 MAX M9, 2026-08-11]

### Host predictors for bootstrap pass one

The predictors were scored off-line from the per-column readout (`ROCKET_RK3576_CALMAP`)
[HW sweep, H96 MAX M9, 2026-08-11]. The readout covers both calibration forwards of
Qwen2.5-1.5B at `M`=2048 and `-ub` 2048. It cannot see perplexity, another model, another `M`,
or decode.

The chunk-to-chunk colmax ratio and the second forward's colmax rise come from the same
readout. The frontend bootstraps each chunk separately because each chunk's accumulator is over
its own contraction. The ratio across the four chunks of one split `ffn_down` agrees with that
reason.

### A row cut for pass one

One process ran per test point, with the `CALMAP` readout on [HW sweep, H96 MAX M9,
2026-08-11].

### Row-cut cost and `BOOTMARGIN`

The 8-window table was measured on 2026-08-12 [HW sweep, H96 MAX M9]. A same-build control ran
at both ends of the first configuration and is identical to every printed digit. The
measurement covers one model, 8 windows of one text and one `-ub`, and no decode.

Paired per window against the 9.9348 control over those 8 windows, `CALROWS`=128 at 2.25 is
-0.0055 +/- 0.0069 (se). The `BOOTMARGIN`=2.25 configuration alone is -0.0047 +/- 0.0076.
Neither is resolvable, so neither is an improvement. Both are 8-window figures, which are
window-set artifacts, and the depth readings below replace them.

#### The row cut and the margin at depth

The sweep is eight test points [HW sweep, H96 MAX M9, 2026-08-21]. One `.so` and one corpus
were md5-gated throughout, and the model was md5-gated on the last two. The ubatch is `-ub` 512
and the governor `ondemand`.

The 8-window readings against the deepest ones:

| Contrast | Model | d=8 | Deepest reading |
|---|---|---|---|
| Paired setting | Llama-3.2-1B | -0.0065 | +0.0009 at d=64 |
| Margin alone | Llama-3.2-1B | -0.0060 | +0.0002 at d=64 |
| Cut alone | Llama-3.2-1B | -0.0050 | -0.0011 at d=64 |
| Paired setting | SmolLM2-1.7B | +0.0017 | -0.0092 at d=29 |
| Paired setting | Qwen2.5-1.5B | -0.0144 +/- 0.0046 | -0.0038 at d=29 |

All five contrasts move 0.005-0.011 nats between d=8 and their deepest reading, and two flip
sign. The d=8 -> d=29 step is ~0.010 nats on three of three models: +0.0091 (Llama), -0.0109
(SmolLM2) and +0.0106 (Qwen). The three agree to within 10% in magnitude and not at all in
sign. A reproducible magnitude with an unpredictable sign is what an artifact of the first six
converged windows looks like. Qwen's recorded 8-window -0.0055 +/- 0.0069 sits 1.07 se from its
own d=8 re-read. The two are consistent, and no cross-build discrepancy lies behind the
difference.

The heterogeneity across the three models is a property of the first two rather than a
measurement of the third. Over the first two models alone tau is 0.0077, and the third model
falls between them. Cochran's Q is 6.55 at its arithmetic minimum, with the third model exactly at the
pooled mean, and 7.3-10.6 anywhere else.

Excluded: damage from frozen-scale clipping, which the 8.04% pass-two saturation figure
suggests. The cut alone is null on the converged windows, so that path does not exist at a
measurable size.

#### The clip fraction as a damage predictor

The margin's two effects are in `ggml_backend_rocket_mul_mat_rk3576()` in `ggml-rocket.cpp`:
it sets pass two's scale `sc2`, and the colmax a saturated column records.

A power law fitted on the five margin-1.5 points, `Dcal = A * clip^1.855`, evaluated at the
margin-2.25 clips, under-predicts the measured `Dcal` by 100.5x (`M`/4) and 13.6x (`M`/16).
Refitting `A` through the `AROW`=1 (`M`/4, 1.5) point puts clip and `Dcal` in the same
activation regime, and makes the miss worse, 318x.

#### Resolution of the returned output

Both are in `ggml_backend_rocket_mul_mat_rk3576()` in `ggml-rocket.cpp`: the calibrating
call's output is dequantized at `sc2`, and `cbound` is set once per weight from `sumabs`.

#### Separability of the calibrating-window damage

Three `g` values and three `h` values, less one scale degeneracy, are five parameters through
six points. The fit therefore has one residual degree of freedom.

Excluded forms for `g`: two smooth two-parameter forms fit its three points. The form
`(1-r)^2.52` in the cut's own undershoot fits to 9.3%, and a log-linear form in the cut fraction
fits to 15.6%. Both are excluded at `M`/2 by sign.

#### The margin factor `h`

The factor was fitted only over margins [1.5, 2.25]. The margin 1.875 is an interpolation
between those anchors, so it separates no extrapolation form. Its own verdict flips on the
`D_res` model, which does not hold below `Q` = 1.5 (§"The row cut at `M`/2"). At the `M`/4 cut,
margin 1.875 sits at `Q` = 1.09 [host-computed]. The two test points at margins 1.25 and 1.00
test the extrapolation.

Excluded: a log-linear form in the margin. It gives `h` 3.6386 at margin 1.00 and 1.9075 at
1.25, missing by 2.02x and 1.35x.

The margin ratio is 50.57 / 30.60 = 1.65x between the two models, and the exponent 4.838 /
4.219 = 1.15x [host-computed]. The 2026-08-22 record gave the 1.65x as the exponent's ratio.

The off-line clip-fraction readout was computed before the two test points ran. Their clip
fractions are 9x larger than 8.7%, the largest the readout had been checked at.

Excluded: the truncation mechanism as a predictor of the damage. Taken literally, it puts
`D(1.0)/D(1.5)` at 21.9-54.2 against a measured 7.31, and at 3.37-3.82 when rescaled by its own
validation miss. Its direction is right, and no magnitude from it is usable.

#### The margin index across architectures and K sets

The test points ran on 2026-08-22 [HW sweep, `-ub` 2048, `--chunks 8`, `rocket` 1.6.0, `.so` md5
`ea40bc1b`, backstop delta 0]. The scoring refuses to proceed unless nine pre-existing `Dcal`
values reproduce to 5e-4 and all three anchors' published R to 0.02.

Two models give each channel two points moving opposite ways. That pattern is compatible both
with one per-model scalar setting both channels and with two independent channels. Llama-3.2-1B
separates the two explanations.

Causes excluded for R:

- Depth and head structure, which move in both pairs of the pair table, the second pair moving
  them the other way
- The tokenizer, the vocabulary, the corpus and the tuning status, which the first pair moves
- The width `n_embd` alone, because R is non-monotone in it (960 -> 13.65, 1536 -> 50.57,
  2048 -> 30.7)

The Llama-3.2-1B `D_res` reading sits 1.5% from the boundary between "tracks the architecture"
and "scales with depth". The body therefore gives the two ratios and not a label.

Excluded: baseline sharpness as a driver. Against PPL[8], `D_res` is non-monotone: Qwen
measures 7.2534 and 0.00811, SmolLM2 7.4265 and 0.02060, and Llama 11.4024 and 0.01626.

SmolLM2-135M was first read, on 2026-08-22, as refuting model scale together with the ratio and
`n_ff` alone. The direct surface of the same day, on which the numerator is monotone in scale,
leaves scale a candidate (§"The direct (cut, margin) surface"). The fixed-depth comparison of the
same day excludes scale as a driver of the level independent of the chunk walk (§"Three
checkpoints at a fixed chunk depth").

#### The denominator of the margin index

The test points ran on 2026-08-22 [HW sweep, `-ub` 2048, `--chunks 8`, `rocket` 1.6.0, `.so` md5
`ea40bc1b`, backstop delta 0]. The scoring refuses unless nine pre-existing `Dcal` values
reproduce to 5e-4 and all four anchors' published `R` to 0.02.

#### The direct (cut, margin) surface

The surface reads five models [HW sweep, `.so` md5 `ea40bc1b`, corpus `3ce70e93`, `rocket` 1.6.0].
The analysis re-runs the same nine-value reproduction check before it reads anything.

Excluded on the numerator: depth, which is non-monotone against it (30, 32 and 24 blocks), its
third exclusion, and `n_ff`/`n_embd`, which differs 1.336x at an identical value.

#### The calibrated-column population

The 3 test points ran on 2026-08-22, and each returned `rc=0` [HW sweep]. They ran at `-ub`
2048 `--chunks` 4, `.so` md5 `ea40bc1b`, corpus `3ce70e93`, backstop 650 -> 650, 0 dmesg WARNINGs,
taint 4608, `ondemand`.

The population closed form, generalized to a per-GEMM chunk count, gives both integers exactly:
total 2666496, minority 443776, share 16.643%. It is exact on 12 of 12 integers over six model
configurations. The split configuration is the first at a chunk count the default path never
produces.

Excluded: the reading that the split's arithmetic swamps the setting. The split's own control
is 0.9937x the unsplit control's perplexity.

The scoring refuses unless eleven already-scored `Dcal` values reproduce to 5e-4.

#### Clip damage at a second chunk size

The three test points ran at `--chunks` 4 and `-ub` 2048, 231-264 s each, with a backstop delta
of 0. The population, 3563520 columns, is a third exact forward integer from the closed form,
which is exact on 14 of 14.

Excluded: a reciprocal law in the chunk count. Its apparent 3.2% fit at `KCHUNK`=512 is the law
agreeing with its own first measurement. The local exponent between the two split test points
is 2.8135, against the 1.0230 that the 512 point alone gives.

Excluded: the reading that a shallower contraction clips less. A shallower chunk clips more,
within a single run as well as between runs.

The scoring refuses unless thirteen already-scored `Dcal` values reproduce to 5e-4.

### Three checkpoints at a fixed chunk depth

[HW sweep, 6 runs `rc=0`, `-ub` 2048 `--chunks` 4, `.so` md5 `ea40bc1b`, corpus `3ce70e93`. Each
run took 95-607 s, backstop 689 -> 689, 0 dmesg WARNINGs, taint 4608, `ondemand`, 2026-08-22.]

The SmolLM2-360M column is the run of §"The calibrated-column population", not re-run.
Each score was gated on reproducing ten already-scored `Dcal` values to 5e-4.

The population closed form gives both new populations exactly: 853248 columns with the `Kc`
histogram {512: 478464, 64: 374784}, and 6684672 columns with {512: 6684672}. With these five
values the form is exact on 16 of 16 integers.

At `KCHUNK`=512 the column-weighted chunk count `C` is 2.0844, 4.0943 and 4.8571, monotone in
scale over exactly these three checkpoints. The comparison therefore cannot separate a
chunk-count driver from a scale driver. The order by scale reverses between the default walks
and `KCHUNK`=512, so the inversion excludes either as a walk-independent driver of the level. It
does not exclude either as a term in the split's effect, which over these three is monotone in
scale and so in `C`. This is the sixth configuration in which the resolution and clip channels
separate.

The comparison was built to tell two outcomes apart, a collapse of the spread and a surviving
spread. The result is neither: the spread widens and its order inverts.

#### Excluded causes

- *A common multiplicative fall.* It predicts that the 1.8719x is preserved, and the order
  inverts.
- *A `1/C` law.* It gives +0.24885 and +0.25701, 1.46x high and 0.53x low.

### The `KCHUNK` sweep on SmolLM2-135M

#### Shares of the fall

From the unsplit `Dcal` of +0.51870, the 2048 setting (`n_embd` split, `ffn_down` whole) falls
0.14229 and the 512 setting (both split) 0.15640. The whole fall is 30.2% of the unsplit level,
the `n_embd`-K family's part 27.4 points of it, 91%, and `ffn_down`'s 2.7 points, 9%. `ffn_down`'s
0.01411 is 3.75% of the 2048 setting's own `Dcal` [host-computed]. The 88% of a 31.1% fall recorded
on 2026-08-22 summed 27.4% of the unsplit level and 3.7% of the 2048 setting's level, two bases.

#### Runs

A five-setting `KCHUNK` series ran on one checkpoint, and `KCHUNK`=320 was added as a sixth
setting. Up to and including `KCHUNK`=320: twelve runs, 92-156 s each, backstop delta 0. Each
score was gated on reproducing seven already-scored `Dcal` values to 5e-4. The `KCHUNK`=320
setting is the first chunk depth here that is not a power of two.

#### Populations and frozen sub-populations

The population closed form is exact on 20 of 20 integers. Four of them are new here, 818688,
1331712, 2288640 and 784128 columns, each with its `Kc` histogram. The `Kc`=64 sub-population
holds the same 374784 columns in all five split runs, and measures `est1` within 1.0054x and clip
within 1.0033x.

The 749568 `n_embd`-side columns are frozen at `512+64` in the 2048, 1024 and 512 runs, and they
hold `est1` to 1.0013x. That cross-tab is exact because the CALMAP has the tensor name in
column 0. It costs nothing, because a ragged walk leaves a sub-population untouched.

The per-`Kc` clip transfer is good to 0.9 absolute, but it is not monotone, since the 128 setting
clips less than the 256 setting:

| setting | 1024 | 256 | 128 |
|---|---|---|---|
| transferred from the 512 setting, % | 84.506 | 84.945 | 86.083 |
| measured clip, % | 84.643 | 85.826 | 85.244 |

At `KCHUNK`=320 the whole-run clip, 81.879%, is below the transfer's 83.540%. The transfer's
per-`Kc` misses there, −2.725 and −0.812 absolute, are its worst.

At `KCHUNK`=576 the population is exact (513024 columns, `Kc` 576:443904 256:34560 128:34560).
The whole-run clip measures 82.538%, against 81.214% from a transfer that substituted the 2048
setting's `Kc`=1536 clip rate for the unsplit run's.

#### Fits to `Dcal`

A law in the maximum chunk depth predicts exactly zero change between `KCHUNK`=2048 and the
unsplit control. It is refuted with no fitted exponent and no argument about which tensors
account for the loss.

The difference between the 1024 and 512 settings is a real measurement of an aggregate, and it
says nothing about the mechanism.

A count power law with the single parameter `p`=0.4886 is fitted at unsplit -> 512. It fits
unsplit -> 2048 (0.7258 measured against 0.7278) and unsplit -> 1024 (0.7149 against 0.7127)
both to 0.3%. It misses 256 and 128 by 1.79x and 3.56x. The boundary falls exactly where `n_embd`
stops walking `512+64`. The exponent is therefore a property of how finely that family is split,
not of the program's chunk count. The segment exponents swing 41x on the depth:

| segment | 1024 -> 512 | 512 -> 256 | 256 -> 128 |
|---|---|---|---|
| exponent on the count | 0.5616 | 1.8023 | 1.7540 |
| exponent on the depth | 0.0335 | 1.1576 | 1.3702 |

At `KCHUNK`=320 the count law gives 0.9626 of the 512 setting's `Dcal`, and the measured fall is
2.06x. The count law is refuted at 1.98x, and the mean-depth reading with it.

#### Fits to the control level

The control's own perplexity level is where the rotation width acts at first order, and both
single-regressor fits fail there. The column-weighted depth and the rotation width correlate
+0.9963 across the five split settings, and decorrelate only at 320. There the depth-only fit
gives 15.3112 and the rotation fit 15.0787, against a measured 15.2165. Those misses are 3.2x
and 5.3x their own worst in-sample residuals. Only a chunk-count fit survives there (15.2061, error +0.0104
against a residual of 0.0255). That fit was made after the 320 measurement.

The chunk-count fit to the control level then fails on a new run. At `KCHUNK`=576 the control
measures 15.2313 against the fit's 15.2897, a miss of 2.3 worst in-sample residuals. The depth
and rotation fits, which it beat after the fact, miss by 7.4 and 10.6 on the same measure. Its
miss is monotone in the extrapolation (0.0255 worst inside the fitted range, −0.0584 at
`C`=1.25328, −0.0955 at `C`=1). It therefore fails as an extrapolation rather than as a fit.

At `KCHUNK`=576, 73.1% of the columns run the unsplit program. There `ffn_down`'s own control
step is +0.0050, against −0.0060 at the 2048 -> 512 pair, the opposite sign.

#### Excluded causes

- *The global maximum chunk depth.* `KCHUNK`=2048 and the unsplit control share it, and `Dcal`
  differs 27.4%.
- *The chunk count and the mean depth.* `KCHUNK`=320 shares both with 512, and `Dcal` differs
  2.06x.
- *The `Kc`=64 chunks.* The 320 setting is the only walk without them. That population is held at
  exactly 374784 in all five other split settings, while `Dcal` spans 5.99x. A fraction reading
  predicts a 0% run below the 128 setting's +0.06283, and the 320 setting sits 2.81x above it.
  Whether removing them contributes to the 2.06x fall is unbounded. No per-column statistic can
  bound a loss share, and this setting has no frozen sub-population.
- *A `D_res` that follows the `ffn_down` split alone.* In the two runs where `ffn_down` is one
  chunk, `D_res` sits at the unsplit level. In the four where it is split, `D_res` collapses to at
  most 0.0078 in absolute value, but with alternating sign. This is an observation over two points
  against four, not a mechanism [hypothesis]. That reading held over five runs, and `KCHUNK`=576
  refutes it. Every one of those five runs split both families, so the reading took a condition of
  two families one family at a time.

### The `ffn_down` chunk-count dose-response

`KCHUNK` 832 and 768 with their controls: six runs, 95-105 s each, backstop delta 0. Each score
was gated on reproducing eight already-scored values to 5e-4, with a second reader for the
frozen-group spread. A further three runs, 90-107 s each, with backstop delta 0, were scored
under the same seven-value gate.

A three-point extrapolation from the first three chunk counts gives a per-chunk step of −0.00560
then −0.00851, apparently growing. It misses the fourth by 0.0324, which is 3.8x the largest
measured step.

The control-level result is measured directly rather than by extrapolation, on one pair at one
`C`. It bounds the fits without measuring a law.

The margin of the both-split ordering falls from 0.0098 to 0.0025 with the 832 and 768 runs.
Both populations are exact (887808 and 818688 columns, `Kc` histograms exact).

### `BOOTMARGIN` and the no-cut response

[HW sweep, 2 new runs `rc=0`, `--chunks` 4, `.so` md5 `ea40bc1b`, backstop 650 -> 650, taint 4608,
0 WARNINGs, 2026-08-22.]

### The row cut at `M`/2

The cut is read back out of the runs' own bytes rather than off the launcher. The `r` on the
clean set of each perplexity process's own CALMAP is 0.94335. That is strictly between the
no-cut 1.0 and the `M`/4 cut's 0.87151, so 1024 rows were used. Both runs are `rc=0` and
`.so`/corpus/model md5-gated, with a backstop delta of 0.

### Width of the `M`/2 gain

Qwen2.5-1.5B at 1536 and 768: [HW sweep, 2 runs, `rc=0`, `.so`/corpus/model md5-gated,
backstop delta 0, 2026-08-22]. SmolLM2-1.7B at the same cut and margin: [HW sweep, 2 runs,
2026-08-22], both `rc=0` and md5-gated, with a backstop delta of 0.

#### Excluded causes

- *An interpolation of the four runs measured before 1536 and 768.* Three interpolations were
  fitted to those four points, in three variables. They are a cubic in `d`, a two-term
  `0.00811 - K d + C d^k` in `d`, and a cubic in ln(`M`/`M1`). No interpolation of them reaches
  either value, and all three get the sign wrong at 1536. There they predict -0.0053, -0.0193
  and -0.0091 against a measured +0.00335, spread 3.6x among themselves, on the wrong side of
  zero from the measurement. The feature they interpolate across is narrower than their own knot
  spacing.
- *The mixed-`M` population.* The minimum survives every reparametrization tried.

### The cut axis on SmolLM2-1.7B

[HW sweep, 3 runs, `rc=0`, `.so`/corpus/model md5-gated, backstop delta 0, 2026-08-22]

The resolution channel's span over the three models is +0.02060 / +0.00811 = 2.54x, with
Llama-3.2-1B's +0.01626 between the two [host-computed]. The 2.00x recorded on 2026-08-22 is
Llama-3.2-1B against Qwen2.5-1.5B alone.

### The mixed-`M` population

Cross-tabbing a control CALMAP by GEMM group, rather than only by `M`, shows what the minority
is. It is not scattered through the model.

[host-computed, derived from the four control CALMAPs and the GGUF headers, and confirmed on a
fifth model, 2026-08-22.]

The SmolLM2-135M control run measured 409344 / 105600.

### Clip-fraction statistics

The run record for this section and the next is under §"The `est2` readback and `Dcal`
reconstruction".

Two controls bound the off-line clip computation. The first is `est1` itself, which is identical
on the clean set (0 of 4096). It drifts on 7.25% of all columns between the sweep's fixed cut
at margin 1.5 and the same cut at 2.25.
The drift concentrates on small-code columns, with a p50 relative move of 1/3, because a
calibrating call's returned output is the next layer's input. The second control re-evaluates
the sweep against the same cut's other `est1` vector. That moves the predicted clip by 0.9% at
margin 1.875, rising to 9.1% at 3.0, where the counts get small.

#### Excluded causes

- *A saturating form in the clip fraction.* The form `D_clip = A*x/(x + x0)` with A = 0.1434
  holds the four runs at `x` <= 4.35 to 12%. Its supremum is A at every `x`, yet `q_r128` sits
  4.86x above it and `q_r32` 14.13x above it. At margin 1.875 it measures 1.65x high.
- *Severity.* The runs `q_r512` and `q_shipfrac` clip by the same relative amount, mean 1.11485
  against 1.11564, 0.07% apart, at 52x different clip counts.
- *The clipped mass, and the depth the mass sits at.* Normalized by clip count, `D_clip` spreads
  13.51x across the three runs with a row cut. Normalized by clipped mass it spreads 6.85x, and
  normalized by mass weighted with the blocks downstream of each clipped column it spreads 6.93x.
  The depth profile of the clipped mass is near-identical between them (22-30% in `blk.0-3`,
  36-47% in `blk.18-27`).

### The `est2` readback and `Dcal` reconstruction

[HW sweep, H96 MAX M9, 2026-08-22. Three CALMAP runs (172-181 s) and one perplexity run (600 s),
`.so`, corpus and model md5-gated, all `rc=0`, backstop delta 0, taint 4608, 0 `dmesg`
WARNINGs. The off-line sweeps took no device time. The two runs that refute the saturating form
ran in an earlier measurement and cost only their CALMAP.]

The controls of that measurement each report a magnitude. The oracle saturates 0 of 1500160
columns. Where a run did not clip, the oracle reproduces its own `est2` to a clean-set p90 of
1.1%, the readback quantum's own size. It reproduces each run's recorded `sat` flag on
99.805-100% of clean-set columns, against a 12.7% count error at depth. The `Dcal`
reconstruction is the fourth control.

#### Earlier readout measurements

The measurements below, from 2026-08-21 and 2026-08-22, back earlier sections of the calibration
study. They are recorded in the order the body gives them, each scope statement and run record
beside the text it followed.

What the measurement cannot see: any model but Qwen2.5-1.5B, any ubatch but 2048, decode, or
the converged column (`Dcal` is the calibrating-window damage). It cannot see the 4 of 197
weights that calibrate at `M`/2 under `llama-perplexity` but not under `llama-bench`. It cannot
see whether `h` keeps its shape outside [1.5, 2.25], and 1.875 is an interpolation in margin
though an extrapolation in `x`. Nor can it see any cut between `M` and `M`/4, where `g` is
unmeasured.

[HW sweep, H96 MAX M9, 2026-08-21. One perplexity run (678 s) and one CALMAP run (219 s), `.so`,
corpus and model md5-gated, both `rc=0`, backstop delta 0, taint 4608, 0 `dmesg` WARNINGs. The
est1 readout took no device time.]

The est1 readout has a control that must succeed. At fixed input `est1` is margin-independent,
measured at 0 of 4096 mismatching rows on the clean set. That set is small by necessity. Of all
rows, 11.7% disagree between two runs that differ only in the margin, both with no row cut.
None of them is in the graph's first weight, and the share rises with depth. The cause is that a calibrating call's
returned output is the next layer's input.

What that measurement cannot see: any model but Qwen2.5-1.5B, any ubatch but 2048, the
`BOOTMARGIN`×`CALROWS` interior, decode, and depth. On depth, the converged column came back
-0.00522 +/- 0.00287 (5 of 6 windows better) against a null at `-ub` 512. The 6 windows cannot
outweigh the 27-62-window runs, so that reading is recorded rather than claimed.

[HW sweep, H96 MAX M9, 2026-08-21. One perplexity run and four CALMAP runs, `.so`, corpus and
model md5-gated, all `rc=0`, backstop delta 0, 0 `dmesg` WARNINGs.]

A readback out of the instrument's own bytes confirms that `BOOTMARGIN` took effect. On saturated
columns, `est2/est1` is 1.506-1.507 at margin 1.5 and 2.257-2.259 at 2.25. That separates "the
margin did not take effect" from "channel one is inert". The scorer reproduces five recorded
values before scoring anything new. What it cannot see: another model (every anchor is
Qwen2.5-1.5B's), another ubatch, the rest of the `BOOTMARGIN`×`CALROWS` interior, or decode. Nor
can it see depth, since 6 converged windows cannot outweigh the 27-62-window runs.

The ubatch does not set `M` uniformly under `llama-perplexity`. Of the 197 weights, 4 calibrate
at `M`/2: `blk.27.ffn_{down,gate,up}` and `token_embd`. They are the last block's FFN and the
output head, which llama.cpp runs on a reduced token set. That holds at both ubatches, so it
dilutes the contrast symmetrically. Under `llama-bench` the `M` column is uniform.

[HW sweep, H96 MAX M9, 2026-08-21. The `.so`, corpus and model were md5-gated, and all runs
returned `rc=0`, with backstop delta 0 and 0 `dmesg` WARNINGs. A `--chunks 4` control reproduces
the `--chunks 8` control's first four windows to four decimals. What it cannot see: another model
(both anchors are Qwen's), another margin, `-ub` 4096, decode, or depth, since 6 converged
windows cannot outweigh the 27-62-window runs.]

### The row cut's saving

[HW sweep, H96 MAX M9, seven `--chunks 4` runs, `setarch -R` confirmed by readback, one
`.so`/corpus/model all md5-gated, backstop delta 0, 2026-08-21.]

The 17.93% saving at `CALROWS`=128 is 18.02% common-mode-corrected, 18.13% drift-corrected and
18.08% on a reversed repeat.

### Calibration with `NCAL`=1

#### Accuracy

On Qwen2.5-1.5B, a readout's objection that 0.061% of columns would be exposed needed roughly 18x
the measured windows-3-8 shift to show. The library's saturation report is fed `NULL` by the
calibrating branch, so its silence bounds only the converged phase.

The 29-window SmolLM2-1.7B run reproduces the 8-window configuration's six per-window
differences exactly, so nothing changed between the two but the sample. The 8-window +1.40% was
set by which windows the text starts with. The 8-window cross-model ratio of 3.9x was off by the
same 4.5x. Every 8-window number on this path has the same limitation, Qwen's own +0.36%
included.

The aggregate is exact rather than noisy. A second control, `ctl2`, reproduces `ctl` at
+0.0000 on every window, on both models.

SmolLM2's `ctl` returned 10.3354 on a different build from the one that first recorded
10.3354, where two Qwen builds spanned 0.28%. That spread is build-specific arithmetic that
reached one model and not the other. It is not a general between-build error bar.

The row cut's -0.0004 +/- 0.0040 over windows 2-8 is the best-resolved number in that
configuration.

#### Window 1 as a control

Window 1 is byte-identical under both settings. `calibrating` is `cal_done < ncal`, 0 is below
both 1 and 2, and the freeze happens after the output is written. Its d(NLL) is +0.000000, which
makes it a control inside every `NCAL` test point. Its own wall-clock time is that process's
noise floor on identical work, measured at 5.4%.

#### Saving from `NCAL`=1

The two configurations' per-call figures agree to 1.4% and 1.8%. The 11.5 s projection against
the measured 11.6 s is a closure of 0.994.

The run's wall-clock time moved the other way (4.38 -> 4.62 min) and carries no signal. Window
1's identical work differed 5.4% between the two processes, and the totals differed 5.5%, so the
pair is entirely inside the process noise floor. The `quant+rot` bucket, which this
configuration cannot reach, moved +5.5% in the same pair.

[HW sweep, H96 MAX M9, 2026-08-12. The profiled `--chunks 4` test points reproduce the
unprofiled `--chunks 8` runs' window-4 cumulative to four digits (8.7163 and 8.7238). So
`ROCKET_MM_PROFILE` does not perturb the output. What the sweep cannot see: one model, 8
windows of one text, one `-ub`, decode, and whether `CALROWS`'s own -28.2% still holds at
`NCAL`=1. That ratio was measured at `NCAL`=2, and its composition here is [expected].]

### The calibration regime

#### Bucket accounting

Before the `calscan` bucket existed, the calibration path left 27% of its time unbucketed at
`M`=512, rising to 48% at `M`=2048. That gap was the bootstrap's per-column maximum scan.

#### Revision history

The calibrating-to-converged ratio per frontend call at `M`=512:

| Measured | Calibration scan | Calibrating call | Converged call | Ratio |
|---|---|---:|---:|---:|
| 2026-08-11 | column-outer | 94.2 ms | 36.4 ms | 2.59x |
| 2026-08-12 | row-major, byte accumulator | 117.01 ms | 58.14 ms | 2.01x |

The 2026-08-11 row is Qwen2.5-1.5B with no K-split, read from the `-ub` sweep of §"Time and
wall-clock share of the offloaded W8A8 path". Its excess over 2.0x was the bootstrap's own host
work, the column-outer scan, which the entry bucket does not include. The 2026-08-12 row is the
`NCAL`=2 configuration of §"Saving from `NCAL`=1", with `AROW`=1 and the K-split on.
[HW sweep, H96 MAX M9, 2026-08-11 and 2026-08-12]

### Accuracy of the W8A8 path inside a model

The CPU configuration ran first and the NPU configuration second, in the same boot. The
block-diagonal Hadamard fallback is a static in `ggml-rocket.cpp`, and no test in
`rocket-userspace` can link it.

### Output scale of the int8 entry

`ggml-rocket/test-hadamard-quant.cpp` builds the orthonormal Hadamard for a non-power-of-two
`K`.

A standalone host micro measured the W8A8 path's host terms on the board's own A72s. It
reports the best of 5 after a discarded first, pinned as the section's Conditions state.

#### Excluded causes

- *Rank-1 concentration of the requant error, as the cause of SmolLM2's 86x under per-axis
  input scales.* A per-tensor requant's error is uniform in accumulator units, so per-axis
  input scales make it rank-1 in real units. On SmolLM2 the configuration does concentrate: up
  to 20.6% of a call's error energy in one row of 512, against 0.62% for per-tensor inputs.
  Qwen's numbers have the same shape, 14.84% against 0.64%, and Qwen does not blow up. Its
  per-row scale spread is 4.4x median and 24x max, against SmolLM2's 3.7x and 22x. The
  concentration statistic does not separate the two models, so it is not the explanation.

### Sources of the per-column output scale

The configurations are the `_cal`, `_bnd` and `_calshuf` suffixes of
`rocket-userspace/tools/rk3576-w8a8-ppl.py`.

#### The random-column control

The last row of each table is the control that makes the frozen scale's cost attributable. It
applies exactly the frozen-scale configuration's multiset of per-column perturbations to a
fixed random permutation of the columns. The magnitude and the distribution are the same, and
nothing remains of which column needed the headroom. On Qwen its profile is, if anything,
harsher (median 1.078x, tightest column 0.209x, saturating 0.326% against the frozen scale's
0.180%). It costs 1.007x against 1.047x. The metric is therefore stable under a same-size
perturbation that holds no information.

### A global safety factor on the frozen scale

The sweep runs one process per point.

### The two bootstrap passes on the device

The device runs are `tests/rk3576_w8a8_route`. A call at `M`=1 returns `rc`=0. A host readout
prints the pass-one shift as 39 over the whole `sum|w|` range the section gives.
[HW sweep, test run, H96 MAX M9, `tests/rk3576_w8a8_route`, 2026-08-11]

The crossed `N`-tile table and the 1.43x figure come from the same test.
[HW sweep, test run, H96 MAX M9, `tests/rk3576_w8a8_route`, 2026-08-11]
