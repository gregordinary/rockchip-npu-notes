# RK3588 validation record

How the facts in [rk3588.md](rk3588.md) were established: the measurements, controls, searched
space and excluded causes behind each section. The sheet states each fact. This record states
its evidence, under the sheet's section names.

## Programming model

### fp16 matmul arithmetic

The scoring ran on the vendor RK1 [HW sweep, vendor RK1, `rknpu` 0.9.8, pinned per core,
`rocket-userspace/tools/npu_core_diff_score.c`, 2026-09-27]. Off lane 0, the bit-for-bit match
covers about 39 million elements on core 0, 22 million on core 1 and 70 million on core 2.

The group order is part of the model: reversing it misses 69% of the random outputs.

### Clock-gating registers

The values swept on the fp16 matmul program:

- `MAC_GATING` written as 0, as its reset value 0x07800800, as all ones (0x07FFFFFF, every bit
  of `SLCG_OP_EN`), and with each of bits 0-26 toggled alone from the reset value.
- `soft_gating` 0x3F and each bit alone.
- `CLK_GATE` 0x17 and each bit alone, and all three registers together.

Their outputs are byte-identical to the unwritten program, except four `MAC_GATING` values
[HW sweep, vendor RK1, 35 calls a core, `rocket-userspace/tools/npu_core_diff_gate.sh`,
2026-09-27]:

- Bit 11 or bit 23 cleared: the job completes. That half of the output lanes writes one
  constant word per core, and the other half is byte-identical.
- Bit 24 cleared, or the register written 0: the job times out on every core. The DPU raises
  no completion and the driver resets the core, and the control is byte-identical afterwards.

The other 24 single-bit toggles are byte-identical: bits 25 and 26 cleared, and each of the 22
bits that are clear at reset set. A write that takes effect and changes nothing looks the same as one
that never takes effect, and `CLK_GATE` has no witness of its own. Not searched: multi-bit
`MAC_GATING` values other than zero, all ones and the reset value, and bits 31:27.

For CORE `0x3030`, all ones and each bit alone were swept [HW sweep, vendor RK1,
`ROCKET_CORE_3030`]. Each is byte-identical to the control on 35 of 35 calls per core.

## Errata

### E7: fp16 matmul output differs between cores on output lane 0

A group *fires* on a core when that core writes an event for it.

#### Lane-0 events

The lane-0 events per core:

| Core | Events over the characterization runs | Events on one re-run set of 33 calls |
|---|---:|---:|
| 0 | 7095 | 3648 |
| 1 | 181 | 160 |
| 2 | 1037915 | 61761 |

On that data every event of core 1's is in core 0's or core 2's set. Of core 0's events, 93%
are in core 2's. The re-run of that set was byte-identical to its first run on every core
[HW sweep, vendor RK1, pinned per core, `tests/npu_core_diff_probe` raw mode, 2026-09-27].

#### Dependence on the inputs

The characterization's groups all derive from one attention dot product. A second run holds
1.05 million groups, built from 64 permuted variants of it [HW sweep, run t2]. On it core 1
fires on 743,682 and core 2 on 578,119. There the sets do not nest. Of core 0's events, 63%
are in core 2's set, and of core 1's, 77% are in core 0's or core 2's.

A re-run of 128 of those calls was byte-identical on every core. It gave 37,315 events on
core 0, 48,449 on core 1 and 35,059 on core 2.

The flash-attention element (256,240) sits on lane 0 and is in core 2's set only. Its identity
copy, placed at all 131072 positions of one call, deviates at 8192 of them on core 2. Those are
lane 0 of every row, and it deviates nowhere on cores 0 and 1. Two RK3588s give byte-identical
outputs per core for its tile [HW sweep, 2026-09-27,
`rocket-userspace/tests/fa_core_matmul_probe`]:

| Board | Driver | How the core was known | Core 2 | Cores 0 and 1 |
|---|---|---|---|---|
| Mainline RK1 | `rocket` 1.3.0 | `gpu_scheduler` trace, per job | `3f4d6f30`, 24 of 24 calls | `3f4d7030`, 66 of 66 |
| Vendor RK1 | `rknpu` 0.9.8 | `RKNPU_CORE_MASK`, pinned | `3f4d6f30`, 10 of 10 | `3f4d7030`, 20 of 20 |

#### The deciding K group

Permuting the tile's 384 K indices, applied to both operands, keeps every exact dot product:

| Permutation | Tiles where (256,240) differs between cores |
|---|---|
| None | 1 of 1 |
| Whole 32-, 64- or 128-wide blocks moved, contents kept | 20 of 20 at each width |
| Products moved only inside their own 8-, 16- or 32-wide block | 10, 9 and 6 of 20 |
| Products moved across 32-wide blocks (whole 8- or 16-wide blocks, inside 64 or 128, any) | 1-2 of 20 |

Every row counts tile 0, which is unpermuted. A move inside the group can put the event in
core 0's set instead. Core 0 was the core below the model in 10 of the 25 in-group cases.

Single transpositions of the deciding group (k 32-63) map the sets, 32 repeats each:

- **Core 2**: 37 of 496 remove the event.
- **Core 0**: 24 create one, and every one of them moves lane 5 (k 37).
- **Core 1**: none moves it.
- **Outside the group**: 0 of 341 transpositions and 0 of 704 knockouts or sign flips move it.

#### The coarse frame

The weight scaling by 16 and by 1/16 is run s13 [HW sweep, vendor RK1, run s13].

Each of 885,667 recorded lane-0 mismatches is one group's event, reproduced bit for bit
[`npu_core_diff_score` with `NPU_CORE_DIFF_ATTRIB=1`]. None is ambiguous between groups.
Repeat runs of s8 on core 2 and of s13 on all three cores gave byte-identical output.

#### Event dependencies

The counts behind the sheet's list [HW sweep, vendor RK1, 2026-09-27]:

- Every product of the group: 32 of 32 single knockouts remove it.
- A zero feature word: 0x0000 anywhere in the group disables it, in 0 of 8188 variants.
- The coarse significands: every re-encoding that keeps each coarse significand keeps core 2's
  event (47 of 47). Of those that keep the product but move bits between the operands, 8 of 38
  move it.
- The lanes below the frame's LSB: randomizing the one frame-aligned product's significand
  keeps the event 40 of 40 times. Randomizing every shifted lane keeps it 0 of 128.
- A rare coincidence among those lanes: 40,960 random zero-free groups fired 0 times on every
  core, including groups with the triggering group's exponents and signs. Sum-level candidates
  (the group sum's fraction, carry counts, shift-class sums) reach precision 0.26 and recall
  0.70 on core 2's held-out half, and at most 0.005 precision on cores 0 and 1.

#### Position classes

Each core's group positions fall into classes. Two positions are in one class when swapping
their contents never moves an event. Each pair was swapped on 750-1500 bases that fire
[HW sweep, vendor RK1, run t2]. A swap across classes moves the event on 1-24% of them:

| Core | Classes of the group's 32 positions | Singletons | Unresolved |
|---|---|---|---|
| 0 | {0-2, 6, 8, 12-15, 19, 22-26, 29}, {16-18, 21, 27, 30}, {11, 20} | 3, 4, 5, 7, 28, 31 | 9, 10 |
| 1 | {3, 5, 6, 9-11, 13-25, 27-30}, {0, 2, 4} | 1, 7, 8, 26, 31 | 12 |
| 2 | {0-4, 6, 10, 16, 17, 19-22, 26-28, 31}, {7, 9} | 5, 8, 11-15, 18, 23, 25, 29, 30 | 24 |

An unresolved position differs from a class on under 3 bases in 1000. The class sizes differ
between cores, so no relabeling of positions maps one core onto another. Swaps give the classes
but not how they nest, so the reduction tree's shape is not recovered.

One lane's content moves the event the same way across a class. One lane was swept over
8192 words at each of the 32 positions [HW sweep, run t3]. On core 2 the event pattern is the
same at every position of the large class and at 8, 13, 18 and 30. The other positions share
three patterns, {5, 7, 14, 15, 25}, {9, 11, 12} and {23, 24, 29}. Core 0 with the lane at
position 5 reproduces core 2's first pattern exactly, and core 1 fires at none.

#### Predicate search

An event needs most of the group. Setting lanes to -0 lowers core 2's firing rate from 0.66
at one lane to 0.01 at sixteen and to zero at 24. A group of the frame lane plus one to four
others fires 0 times in 262,144 on every core [HW sweep, run t4]. A re-run was
byte-identical. No small firing group exists to decode.

No predicate was found. Candidates were scored on held-out firing groups against the
always-true predicate, whose F1 is 0.724 at core 2's 0.567 base rate. None scored above it:

- Thresholds on the group residue and on the lanes' summed sub-frame fractions.
- Carry counts over eight partial-product decompositions, NVDLA's Booth rows among them.
- NVDLA's carry-save split, emulated bit-exactly from `nvdla/hw`.

An additive model over each lane's content and position class predicts held-out arrangements
of the same bases at 0.996-0.998. Scored on other bases it falls to 0.79-0.95, so it describes
the bases, not the rule.

Every lane content came from one family: Cmax 7, shifts 0-5, 562 contents. The search could
not have found a lane term outside that family. Such a term differs at other frames, at
shifts 6 and 7, or on negative features. The per-lane partial-product split is this silicon's
own, and a sweep cannot observe it.

#### Integer path

Full-range int8 tiles through `rocket_matmul_int8` (int32 out) at the same shape are exact on
all three cores, 2.6 million elements each. The three cores' outputs are identical [HW sweep,
vendor RK1, pinned per core, 2026-09-27].

#### Core placement

The flash-attention handler's two output states are this effect. Over 1800 traced trials, every
odd output had head 12's AV job on core 2. Every other output had it on core 0 or 1. The counts
are 456 of 456 odd, and 1344 of 1344 modal. One worker never reaches core 2, because the
scheduler alternates one fd between cores 0 and 1 [HW sweep, mainline RK1,
`rocket-userspace/tools/fa_trace_join.py`, 2026-09-27].

The DRM scheduler's code accounts for the alternation [source-confirmed: Linux v7.1
`drm_sched_entity_select_rq`, `drm_sched_pick_best`]. An fd's entity changes core only when it
has no job queued and its last job has finished. It then moves to the core with the lowest
score, the first in index order on a tie. The entity's own place on its current core's run
queue counts toward that core's score.

#### Excluded causes

- *A relabeling of lanes between cores.* The search covered 31 rotations, 15 in-half and 7
  in-quarter rotations, and all 120 bit permutations of the lane index. It also covered every
  index XOR, and none maps core 2 onto core 0.
- *The clock-gating registers.* The sweep is under
  [Clock-gating registers](#clock-gating-registers), above.
