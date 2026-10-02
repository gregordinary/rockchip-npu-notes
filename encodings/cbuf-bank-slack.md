# CBUF bank slack for the int8 feature DMA

A matmul or conv tile must declare how many of the 12 CBUF banks its input feature
occupies (`DATA_BANK`) and its weight occupies (`WEIGHT_BANK`). The two partition the 12
banks. The obvious value is `data_bank = ceil(feature_bytes / 32 KB)`: exactly
enough banks to hold the cube. **For the int8 feature cube that is one bank too few**
[HW sweep].

## Symptom

At certain `(Mtile, Ktile)` tile geometries an int8 matmul (`gen_matmul_int8`, run as a
1×1 conv) returns garbage in the last few output rows of the tile. Every column of those
rows is wrong, with values ~1e5-1e6 against the correct ones. Everything else in the tile
is bit-exact. The host int64 K-accumulation is exact, so the fault is the CNA
feature-input DMA reading the tail rows of the cube from the wrong place. The DMA runs one
bank past the `ceil` allocation, and the last rows fall off the end.

## Resonant tile geometries

The bad set is a joint 2-D `(Mtile, Ktile)` resonance with no closed form. The bad
K-groups depend on Mtile and vice versa:

| Mtile | bad K-groups (`K/32`) |
|---|---|
| 144 | 7, 21, 35 |
| 192 | 5, 21, 37 |
| 240 | 17, 21, 25, 29 |

`feature_bytes = Mtile·Ktile`. The resonance clusters where the cube nearly fills its last
bank, but "near-full" is necessary, not sufficient: an exactly-full bank is fine, and
98%-full corrupts. Do not try to predict the set arithmetically. Give the slack.

When the matmul fans N across cores, the resonance looks like a different bug class,
"nt-nondeterminism". The worker count changes the per-worker `Nt`, which changes `Kt`,
which changes whether an emitted tile lands on a resonant `(Mtile, Ktile)`. The root cause
is the same.

## fp16 immunity

An fp16 control at the identical `(Mtile=144, Ktile=672)` tile is bit-exact while int8
corrupts. The fp16 feature cube is C2=8, 2-byte, and the int8 cube is C2=16, 1-byte (see
[tile-layouts.md](tile-layouts.md)). The descriptor stride and bank math (`line_stride`,
`surf_stride`, `data_entries`) is identical for both, sharing fp16's exact-fit bank sizing.
fp16 happens not to resonate. For the same `(M,K)` its 2x byte size lands the last bank at
a safe fill, so the corruption is int8-cube-specific. So the defect stays latent on the
heavily used fp16 path.

## The fix

Reserve one slack bank for the int8 feature: `data_bank = min(fd_banks + 1, 11)`,
`weight_bank = 12 − data_bank`. The weight always has room: its per-kernel bytes are <= one
bank, and it never needs all the leftover. The host tiler must reserve the matching bank,
so that it never picks a tile whose feature cannot get its `+1`. **Budget feature+weight to
11 banks, not 12** (`I8_BUDGET = CBUF_BANKS − 1`). The cost is zero, since for real model
shapes `Kt` is unchanged. Only feature tiles already near 11 banks shrink by one K-step.

A `ROCKET_I8_FDBANK_EXTRA` sentinel sweep on the resonant tile establishes the fix
[HW sweep]. `+1` is bit-exact, `−1` is far worse, and overriding
`surf_stride`/`feature_grains`/`data_entries` instead does not help. The fix is bit-exact
across a full `(M, K, N)` grid (86 failing cases -> 0) and end-to-end (EfficientDet nt=1 ≡
nt=4).

## The int8 and uint8 conv path

The int8 and uint8 conv path is affected too [HW sweep]. The conv path is susceptible to
the same resonance under two conditions. Its generator uses the same exact-fit
`data_bank = ceil(feat_bytes/bank)`, and its direct-conv tiler budgets feature+weight to
all 12 banks with no slack. The generator is `gen_conv2d_int8_fill` (shared by
`gen_conv2d_int8` and `gen_conv2d_dw_int8`), and the tiler is `conv2d_int8_run`. Without
the slack bank the conv path is safe by luck, not by design.

When `datain_width = 1`, the conv feature DMA descriptor matches the matmul's exactly
(`gen_matmul_int8` sets `datain_width=1, datain_height=M, datain_channel=K`). A single-job
int8 1×1 conv at `IW=1`, whose feature cube is the matmul's, shows the identical signature
[HW sweep]. At `IC=1184`, `IH=189..193` and `216..221` (97.6-99.8 % of the last bank), the
last output rows garble (~3e5-9e5) and all else is bit-exact. The output is clean at
<=97 % fill, the same sharp near-full threshold. `+1` clears the whole window (1494/1494
shapes bit-exact), and `-1` worsens it. `IW>=2` shapes (a different descriptor) do not
resonate, which is why real detectors, all `IW≫1`, never hit it.

The fix mirrors the matmul:

- `gen_conv2d_int8_fill`: `data_bank = min(fd_banks + 1, 11)`, `weight_bank = 12 −
  data_bank` (shared by direct and depthwise).
- `conv2d_int8_run` (direct tiler): reserve the matching bank,
  `feat_budget = (12 − 1 − weight_banks)·32 KB`, int8 feature ceiling `CONV_FEAT_BUDGET_I8
  = 7` banks (fp16 stays 8, it is immune).
- **Depthwise** (`conv2d_dw_int8_run`) needs no tiler change. Its weight is one per-channel
  `KH·KW·G`-byte cube (<= 1 bank) and its feature is capped at 8 banks. So the shared
  generator `+1` always fits (`data_bank <= 9`, >= 3 banks left for the weight).

The gate is `tflite-rocket/tests/conv_bank_slack.c`, the conv analog of `mm_nt_det.c`. The
default `IW=1` hot-K sweep passes, and `ROCKET_CONV_I8_FDBANK_EXTRA=-1` reverts to
exact-fit and fails. These regressions are clean:

- `convert_test`
- `conv2d_int8_rocket`
- `conv2d_fp16_rocket`
- `conv_dw_int8_runtime`
- `mm_nt_det GRID=1`

The `ROCKET_CONV_I8_FDBANK_EXTRA` sentinel is kept, relative to the `+1` base, for future
reverse-engineering.
