# F16 weight residency and projection fusion

The host must scatter row-major weights into the native `(N/16,K/32,16,32)` weight cube before
the NPU can consume them (packB). There is no on-chip layout conversion, so that scatter is
irreducible. The streaming matmul path pays it per micro-batch: it re-scatters the same
attention/FFN weight on every call. For a long prefill that repeated packB is a large fraction
of fp16 wall time. Two independent levers each remove a different repeated pack, and they
stack:

- **Residency** (`ROCKET_F16_RESIDENT`) packs an all-K F16 weight once into resident cube
  tiles and reuses it across every later micro-batch and turn. That removes the per-call
  packB.
- **Projection fusion** runs a group of matmuls that share the same input `A[M,K]` (Q\|K\|V,
  or gate\|up) as one combined-N matmul. That removes the redundant per-node packA: the shared
  input is scattered once for the group, not once per member. It also collapses the group's
  submits (5 fusable matmuls per layer become 2).

The two levers are orthogonal: one attacks the weight scatter, the other the input scatter
and dispatch count. Holding a fusable group resident as one combined-N weight captures both
at once.

## Measurement

Warm pp512 and pp2048 ran through ggml-rocket / llama.cpp on an idle RK1, as interleaved
clock-fair pairs [HW sweep 2026-07-15, Llama-3.2-3B-F16, 600 MHz, warm, interleaved]:

| config | pp512 (t/s) | pp2048 (t/s) |
|---|---|---|
| streaming default (fused) | 55.3 | 40.9 |
| resident, no fusion | 61.9 | 44.0 |
| resident + fusion | 65.6 | 46.5 |

The deltas between those configs are:

| comparison | pp512 | pp2048 |
|---|---|---|
| Residency over the streaming default | +11.8% | +7.6% |
| Fusion added to residency | +5.7% | +5.7% |
| Both stacked over the streaming default | +18.5% | +13.6% |

The per-pair range of the fusion delta is +5.5-6.0% at pp512 and +4.5-7.0% at pp2048. Both
pairs are positive at both lengths.

The win itself proves engagement. A silent fallback to the streaming-fused path would read
~55 / 41, which is below resident-no-fusion, not above it. Greedy output is byte-identical
across all three configs (llama-simple, ~300-token prompt).

## Fusion under residency

Fusion's absolute saving is the eliminated redundant packA plus the collapsed submits. It is
roughly 1.3 ms/token at pp512 and 1.6 ms/token at pp2048 on this model. That saving is
independent of packB, so it remains once residency has removed the weight scatter. The two
levers cut different costs, and their deltas compound.

## The mechanism

Under `ROCKET_F16_RESIDENT`, the backend concatenates a fusable group into one `[Ntot,K]` host
buffer and packs it once into a resident combined-N weight. It caches that weight under a
composite key (the members' stable weight names joined by `|`) and reuses it through the
prepacked matmul. It splits the combined `[M,Ntot]` output back into each member's
destination.

One equivalence avoids a new primitive. Packing a host-concatenated `[Ntot,K]` weight is
byte-identical to the streaming segment packer, since both scatter into the same
`(N/16,K/32,16,32)` tiles. So the resident combined weight is bit-exact with no new code path.
Resident bytes equal the sum of the members' bytes, because the concat buffer is transient
and is freed after the pack. The combined weight takes no extra RAM over holding the members
resident individually.

The group runner tries the resident-fused path first. On decline it falls back cleanly to
streaming-fused, with nothing written. The decline cases are a small one-shot `M` below the
tile cap, a group over the resident budget, and a reached IOVA or `MemAvailable` floor. The
entry is `ggml_backend_rocket_mul_mat_group_resident` (ggml-rocket backend only).

## Scope

- **Separate-projection architectures only.** Llama, Qwen and Mistral carry distinct Q/K/V
  and gate/up weights. Architectures that already pack qkv / gate-up as a single combined
  weight (e.g. Phi-4-mini) have nothing to fuse. They are the pure-residency case and gain
  only residency's packB-once.
- **F16 only.** The quantized fused-group extension is a settled negative: a quant
  expert/projection route is per-micro-batch dequant-bound, so fusing it does not pay.
- **Opt-in**, through `ROCKET_F16_RESIDENT` like the rest of the resident family. The default
  (knob off) path is untouched and bit-identical. Decode is unaffected (the source GGUF stays
  mapped). RAM cost is ~2x the fp16 model (resident tiles plus the source), so it wants a
  model that fits ~2x in RAM.

## Residency outcome report

An A/B on t/s cannot tell a residency arm that was declined from one that was placed and
gained nothing. Both produce the same rows at the same speed, and the init line reports only
the *budget* the route was given. So both routes report their outcome at teardown, on the
driver log channel. Set `ROCKET_LOG_STDERR=1` to see it under a host that silences ggml, which
`llama-bench` does without `-v`:

```
[f16-resident] weights offered to the resident route: 193 resident on the NPU (5232MB), 0 streamed
               via the per-call pack -- 100% resident
[f16-resident] admission first declined at 1008MB resident: the 1024MB resident-weight budget
               would not hold the next 96MB group
```

The second line names which of the three admission limits turned a weight away first. The
limits are the byte budget, the `MemAvailable` floor, and a full NPU IOVA window. Each takes
a different fix.
Under 80% resident, a warning says that what was measured is mostly the streaming path.

A fused group's members count individually, so the number means the same thing on both
routes. Llama-3.2-3B-F16 at `ROCKET_F16_RESIDENT=auto` reads 193 resident / 5232 MB with
fusion on and 193 / 5232 MB with `ROCKET_NO_FUSE=1` [HW sweep 2026-08-28, RK1, 600 MHz].

The report does not score three things:

- Whether residency paid. That is the t/s beside it.
- The *work* held resident. The report counts weights and bytes, not GEMM share.
- Anything about matmuls the backend never claimed. The denominator is the weights offered
  to the residency route, so an op refused upstream by type, shape or `rocket_min_m` appears
  in neither column.

A model whose `K` never exceeds 2048 never forms a fusable group, so a small model exercises
only the per-node route.

The probe is a warm pp512 / pp2048 A/B through ggml-rocket / llama.cpp with
`ROCKET_F16_RESIDENT=auto` vs unset. The `ROCKET_NO_FUSE=1` arm isolates the fusion delta at
fixed residency. Read the teardown line on each arm before quoting a delta as a zero.
