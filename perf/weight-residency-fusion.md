# F16 weight residency and projection fusion — two orthogonal pack-once levers

The host must scatter row-major weights into the native `(N/16,K/32,16,32)` weight cube before the
NPU can consume them (packB), and there is no on-chip layout conversion, so that scatter is
irreducible. On the streaming matmul path it is paid **per micro-batch**: the same attention/FFN
weight is re-scattered on every call, and for a long prefill that repeated packB is a large fraction
of fp16 wall time. Two independent levers each remove a different repeated pack, and they **stack**.

- **Residency** (`ROCKET_F16_RESIDENT`) packs an all-K F16 weight **once** into resident cube tiles
  and reuses it across every later micro-batch and turn — it removes the per-call **packB**.
- **Projection fusion** runs a group of matmuls that share the same input `A[M,K]` (Q\|K\|V, or
  gate\|up) as one combined-N matmul — it removes the redundant per-node **packA** (the shared input
  is scattered once for the group, not once per member) and collapses the group's submits (5 fusable
  matmuls per layer become 2).

These are orthogonal: one attacks the weight scatter, the other the input scatter and dispatch
count. Holding a fusable group resident as **one combined-N weight** captures both at once.

## The measurement [HW sweep 2026-07-15, Llama-3.2-3B-F16, 600 MHz, warm, interleaved]

Warm pp512 / pp2048 through ggml-rocket / llama.cpp on an idle RK1, interleaved clock-fair pairs:

| config | pp512 (t/s) | pp2048 (t/s) |
|---|---|---|
| streaming default (fused) | 55.3 | 40.9 |
| resident, no fusion | 61.9 | 44.0 |
| **resident + fusion** | **65.6** | **46.5** |

- Residency over the streaming default: **+11.8% pp512 / +7.6% pp2048**.
- Fusion **on top of** residency: **+5.7% pp512 / +5.7% pp2048** (per-pair range +5.5–6.0% pp512,
  +4.5–7.0% pp2048; both pairs positive at both lengths).
- Both stacked over the streaming default: **+18.5% pp512 / +13.6% pp2048**.

Engagement is proven by the win itself: a silent fallback to the streaming-fused path would read
~55 / 41 (below resident-no-fusion), not above it. Greedy output is **byte-identical** across all
three configs (llama-simple, ~300-token prompt).

## Why fusion carries onto the resident regime

Fusion's absolute saving is the eliminated redundant packA plus the collapsed submits — roughly
1.3 ms/token at pp512 and 1.6 ms/token at pp2048 on this model. That saving is **independent of
packB**, so it does not evaporate once residency has removed the weight scatter: the two levers cut
different costs and their deltas compound. Before this, `ROCKET_F16_RESIDENT` disabled fusion and
sent every weight to the per-node resident path, leaving fusion's saving on the table.

## The mechanism — no driver change, bit-exact by construction

Under `ROCKET_F16_RESIDENT`, a fusable group is concatenated into one `[Ntot,K]` host buffer and
packed **once** into a resident combined-N weight, cached under a composite key (the members' stable
weight names joined by `|`) and reused via the prepacked matmul; the combined `[M,Ntot]` output is
split back into each member's destination. The key that avoids a new primitive: **packing a
host-concatenated `[Ntot,K]` weight is byte-identical to the streaming segment packer** — both
scatter into the same `(N/16,K/32,16,32)` tiles — so the resident combined weight is bit-exact with
no new code path. Resident bytes equal the sum of the members' bytes (the concat buffer is transient,
freed after the pack); no extra RAM over residenting them individually.

The group runner tries the resident-fused path first and falls back **cleanly** to streaming-fused
(nothing written) on decline: a small one-shot `M` below the tile cap, over the resident budget, or
the IOVA / `MemAvailable` floor reached. `ggml_backend_rocket_mul_mat_group_resident` (ggml-rocket
backend only).

## Scope

- **Separate-projection architectures only** — Llama / Qwen / Mistral carry distinct Q/K/V and
  gate/up weights. Architectures that already pack qkv / gate-up as a **single** combined weight
  (e.g. Phi-4-mini) have nothing to fuse; they are the pure-residency case and gain only residency's
  packB-once.
- **F16 only.** The quantized fused-group extension is a settled negative — a quant expert/projection
  route is per-µbatch dequant-bound, so fusing it does not pay.
- **Opt-in**, via `ROCKET_F16_RESIDENT` like the rest of the resident family; the default (knob off)
  path is untouched and bit-identical. Decode is unaffected (the source GGUF stays mapped). RAM cost
  is ~2× the fp16 model (resident tiles plus the source), so it wants a model that fits ~2× in RAM.

Probe: warm pp512 / pp2048 A/B through ggml-rocket / llama.cpp with `ROCKET_F16_RESIDENT=auto` vs
unset, and `ROCKET_NO_FUSE=1` to isolate the fusion delta at fixed residency.
