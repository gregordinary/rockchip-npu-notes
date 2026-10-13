# F16 weight residency and projection fusion

F16 weight residency and the fusion of projections that share an input raise prefill
throughput together. On Llama-3.2-3B-F16 the two together measure 1.081x at pp2048 over the
streaming default [HW sweep 2026-08-29]. The tables below split the gain on the build of
2026-07-15, whose streaming default was slower. There the two gave +18.5% at pp512 and +13.6%
at pp2048 [HW sweep].

Residency (`ROCKET_F16_RESIDENT`) removes the per-call weight scatter, and projection fusion
removes the per-node input scatter. The two gains combine. Greedy output is byte-identical
across all three configurations (llama-simple, ~300-token prompt).

**Conditions:** RK3588 (RK1), NPU clock 600 MHz, Llama-3.2-3B-F16, warm pp512 and pp2048
through ggml-rocket in llama.cpp, idle board, 2026-07-15. The 1.081x is pp2048 at the default
`-ub`, CPU governor `performance`, three interleaved passes with the configuration order
rotated, 2026-08-29 ([data/tuning-matrix.md](data/tuning-matrix.md)).

**Not characterized:** the fusion gain on models other than Llama-3.2-3B-F16, and the split
between residency and fusion on the build of 2026-08-29.

| config | pp512 (t/s) | pp2048 (t/s) |
|---|---|---|
| streaming default (fused) | 55.3 | 40.9 |
| resident, no fusion | 61.9 | 44.0 |
| resident + fusion | 65.6 | 46.5 |

The deltas between those configurations are:

| comparison | pp512 | pp2048 |
|---|---|---|
| Residency over the streaming default | +11.8% | +7.6% |
| Fusion added to residency | +5.7% | +5.7% |
| Both combined over the streaming default | +18.5% | +13.6% |

## Cause

The host must scatter row-major weights into the native `(N/16,K/32,16,32)` weight cube before
the NPU can consume them (packB). There is no on-chip layout conversion, so that scatter is
irreducible. The streaming matmul path performs it on every micro-batch: it re-scatters the
same attention or FFN weight on every call. For a long prefill, the repeated packB is a large
fraction of fp16 wall-clock time. Two independent optimizations each remove a different
repeated pack:

- **Residency** (`ROCKET_F16_RESIDENT`) packs an all-K F16 weight once into resident cube
  tiles, and reuses it across every later micro-batch and turn. It removes the per-call
  packB.
- **Projection fusion** runs a group of matmuls that share the same input `A[M,K]` (Q\|K\|V,
  or gate\|up) as one combined-N matmul. It removes the redundant per-node packA: the shared
  input is scattered once for the group, not once per member. It also reduces the group's
  submits (5 fusable matmuls per layer become 2).

The two optimizations are independent: one removes the weight scatter, the other the input
scatter and the dispatch count. A fusable group held resident as one combined-N weight gets
both at once.

### Fusion under residency

Fusion's absolute saving is the eliminated redundant packA plus the reduced submits. It is
roughly 1.3 ms/token at pp512 and 1.6 ms/token at pp2048 on this model. The fusion saving is
independent of packB, so it remains once residency has removed the weight scatter. The two
optimizations cut different costs, and their deltas compound.

## Resident fused-group path

Under `ROCKET_F16_RESIDENT`, the backend concatenates a fusable group into one `[Ntot,K]` host
buffer and packs it once into a resident combined-N weight. It caches that weight under a
composite key (the members' stable weight names joined by `|`) and reuses it through the
prepacked matmul. It splits the combined `[M,Ntot]` output back into each member's
destination.

One equivalence avoids a new primitive. Packing a host-concatenated `[Ntot,K]` weight is
byte-identical to the streaming segment packer, since both scatter into the same
`(N/16,K/32,16,32)` tiles. The resident combined weight is therefore bit-exact with no new
code path. Resident bytes equal the sum of the members' bytes, because the concat buffer is
transient and is freed after the pack. The combined weight takes no extra RAM over holding the
members resident individually.

The group runner tries the resident-fused path first. If the path declines, the runner falls
back cleanly to streaming-fused, with nothing written. The decline cases are a small one-shot
`M` below the tile cap, a group over the resident budget, and a reached IOVA or `MemAvailable`
minimum. The entry is `ggml_backend_rocket_mul_mat_group_resident` (ggml-rocket backend only).

## Applicability

- **Separate-projection architectures only.** Llama, Qwen and Mistral have distinct Q/K/V
  and gate/up weights. Architectures that already pack qkv / gate-up as a single combined
  weight (e.g. Phi-4-mini) have nothing to fuse. They are the pure-residency case, and gain
  only residency's packB-once.
- **F16 only.** A quantized fused group gives no gain: a quantized expert or projection path
  is bound by per-micro-batch dequantization, so fusing it does not help.
- **Opt-in**, through `ROCKET_F16_RESIDENT` like the rest of the resident family. The default
  path, with the option off, is untouched and bit-identical. Decode is unaffected (the source
  GGUF stays mapped). The RAM cost is ~2x the fp16 model (resident tiles plus the source), so
  the model must fit ~2x in RAM.

A model whose `K` never exceeds 2048 never forms a fusable group, so a small model exercises
only the per-node path.

## Residency outcome report

Both residency paths report their outcome at teardown, on the driver log channel. A t/s
comparison alone cannot tell a declined residency configuration from one that was placed and
gained nothing. Both produce the same rows at the same speed, and the init line reports only
the *budget* the path was given. Set `ROCKET_LOG_STDERR=1` to see the report under a host that
silences ggml, which `llama-bench` does without `-v`:

```
[f16-resident] weights offered to the resident route: 193 resident on the NPU (5232MB), 0 streamed
               via the per-call pack -- 100% resident
[f16-resident] admission first declined at 1008MB resident: the 1024MB resident-weight budget
               would not hold the next 96MB group
```

The second line names which of the three admission limits turned a weight away first. The
limits are the byte budget, the `MemAvailable` minimum, and a full NPU IOVA window. Each takes
a different fix. Under 80% resident, a warning says that what was measured is mostly the
streaming path.

A fused group's members count individually, so the number means the same thing on both paths.
Llama-3.2-3B-F16 at `ROCKET_F16_RESIDENT=auto` reports 193 resident / 5232 MB with fusion on
and 193 / 5232 MB with `ROCKET_NO_FUSE=1` [HW sweep 2026-08-28, RK1, 600 MHz].

The report does not measure three things:

- Whether residency gave a gain. The t/s beside it shows that.
- The *work* held resident. The report counts weights and bytes, not GEMM share.
- Anything about matmuls the backend never claimed. The denominator is the weights offered
  to the residency path, so an op refused upstream by type, shape or `rocket_min_m` appears
  in neither column.

> [!NOTE]
> Read the teardown line of each configuration before quoting a delta as a zero.

## Evidence

### Measurement method

The configurations ran as interleaved clock-fair pairs, warm, on an idle RK1. The measurement
is a warm pp512 / pp2048 comparison through ggml-rocket / llama.cpp, with
`ROCKET_F16_RESIDENT=auto` against unset. A `ROCKET_NO_FUSE=1` configuration isolates the
fusion delta at fixed residency.

The per-pair range of the fusion delta is +5.5-6.0% at pp512 and +4.5-7.0% at pp2048. Both
pairs are positive at both lengths.

The gain itself shows that residency engaged. A silent fallback to the streaming-fused path
would measure ~55 / 41, which is below resident-no-fusion, not above it.

### Revision history

An earlier measurement on 2026-07-15 preceded the interleaved pairs above. It compared
residency without fusion against the streaming-fused default, on Llama-3.2-3B-F16 at 600 MHz
and `-ub 512`. It measured 54.5 -> 59.6 t/s at pp512 (+9.5%) and 39.8 -> 42.1 t/s at pp2048
(+5.9%) [HW sweep 2026-07-15]. The body's figures are the interleaved pairs.
