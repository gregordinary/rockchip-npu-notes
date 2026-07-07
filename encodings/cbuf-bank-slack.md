# The int8 feature DMA needs one CBUF bank of slack

A matmul/conv tile must declare how many of the 12 CBUF banks its **input feature**
occupies (`DATA_BANK`) versus its **weight** (`WEIGHT_BANK`); they partition the 12
banks. The obvious value is `data_bank = ceil(feature_bytes / 32 KB)` — exactly enough
banks to hold the cube. **For the int8 feature cube that is one bank too few.** [HW sweep]

## The symptom

At certain `(Mtile, Ktile)` tile geometries an int8 matmul (`gen_matmul_int8`, run as a
1×1 conv) returns **garbage in the last few output rows** of the tile — all columns of
those rows, values ~1e5–1e6 vs correct. Everything else in the tile is bit-exact. The
host int64 K-accumulation is exact, so this is purely the **CNA feature-input DMA**
reading the tail rows of the cube from the wrong place — it runs **one bank past** the
`ceil` allocation and the last rows fall off the end.

## It is a 2-D resonance, not a clean rule

The bad set is a joint `(Mtile, Ktile)` resonance with **no closed form**. The bad
K-groups depend on Mtile and vice-versa:

| Mtile | bad K-groups (`K/32`) |
|---|---|
| 144 | 7, 21, 35 |
| 192 | 5, 21, 37 |
| 240 | 17, 21, 25, 29 |

(`feature_bytes = Mtile·Ktile`; the resonance clusters where the cube nearly fills its
last bank, but "near-full" is necessary, not sufficient — an *exactly*-full bank is
fine, 98%-full corrupts. Don't try to predict it arithmetically; just give the slack.)

It only looks like a different bug class — "nt-nondeterminism" — when the matmul fans N
across cores: worker count changes the per-worker `Nt`, which changes `Kt`, which
changes whether an emitted tile lands on a resonant `(Mtile, Ktile)`. Same root cause.

## fp16 is immune — the resonance is int8-cube-specific

The fp16 feature cube is **C2=8, 2-byte**; the int8 cube is **C2=16, 1-byte** (see
[tile-layouts.md](tile-layouts.md)). The descriptor stride/bank math
(`line_stride`, `surf_stride`, `data_entries`) is *identical* for both, sharing fp16's
exact-fit bank sizing. fp16 happens not to resonate — for the same `(M,K)` its 2× byte
size lands the last bank at a safe fill — so the corruption is **int8-cube-specific**.
A fp16 control at the identical `(Mtile=144, Ktile=672)` tile is bit-exact while int8
corrupts. This is why the bug stays latent on the heavily-used fp16 path.

## The fix

Reserve one slack bank for the int8 feature: `data_bank = min(fd_banks + 1, 11)`,
`weight_bank = 12 − data_bank`. The weight always has room (its per-kernel bytes ≤ one
bank; it never needs all the leftover). The host tiler must reserve the matching bank so
it never picks a tile whose feature can't get its `+1` — budget feature+weight to
**11 banks, not 12** (`I8_BUDGET = CBUF_BANKS − 1`). Cost: zero — for real model shapes
`Kt` is unchanged; only feature tiles already near 11 banks shrink by one K-step.

Established by an `ROCKET_I8_FDBANK_EXTRA` sentinel sweep on the resonant tile: `+1` →
bit-exact, `−1` → far worse, and overriding `surf_stride`/`feature_grains`/`data_entries`
instead does not help. Validated bit-exact across a full `(M, K, N)` grid (86 failing
cases → 0) and end-to-end (EfficientDet nt=1 ≡ nt=4). [HW sweep]

## The int8/uint8 *conv* path is affected too [HW sweep]

The int8 **conv** generator (`gen_conv2d_int8_fill`, shared by `gen_conv2d_int8` /
`gen_conv2d_dw_int8`) uses the **same exact-fit** `data_bank = ceil(feat_bytes/bank)`, and
the direct-conv tiler (`conv2d_int8_run`) budgets feature+weight to all 12 banks with **no
slack** — so the conv path is susceptible to the same resonance, safe **by luck, not by
design**, unless it reserves the slack bank.

The conv feature DMA descriptor matches the matmul's exactly when `datain_width = 1`
(`gen_matmul_int8` sets `datain_width=1, datain_height=M, datain_channel=K`). A single-job
int8 1×1 conv at **`IW=1`** (so the feature cube is the matmul's) shows the identical
signature — e.g. `IC=1184`, `IH=189..193` and `216..221` (97.6–99.8 % of the last bank):
the **last output rows garble** (~3e5–9e5), all else bit-exact; clean at ≤97 % fill (the
same sharp near-full threshold). `+1` clears the whole window (1494/1494 shapes bit-exact),
`-1` worsens it. `IW≥2` shapes (a different descriptor) do **not** resonate — which is why
real detectors, all `IW≫1`, never hit it. [HW sweep]

The fix mirrors the matmul:
- `gen_conv2d_int8_fill`: `data_bank = min(fd_banks + 1, 11)`, `weight_bank = 12 −
  data_bank` (shared by direct **and** depthwise).
- `conv2d_int8_run` (direct tiler): reserve the matching bank —
  `feat_budget = (12 − 1 − weight_banks)·32 KB`, int8 feature ceiling `CONV_FEAT_BUDGET_I8
  = 7` banks (fp16 stays 8, it is immune).
- **Depthwise** (`conv2d_dw_int8_run`) needs no tiler change: its weight is one per-channel
  `KH·KW·G`-byte cube (≤ 1 bank) and its feature is capped at 8 banks, so the shared
  generator `+1` always fits (`data_bank ≤ 9`, ≥ 3 banks left for the weight).

Gate: `tflite-rocket/tests/conv_bank_slack.c` (the conv analog of `mm_nt_det.c`; default
`IW=1` hot-K sweep PASSes, `ROCKET_CONV_I8_FDBANK_EXTRA=-1` reverts to exact-fit and FAILs).
Regressions clean: `convert_test`, `conv2d_int8_rocket`, `conv2d_fp16_rocket`,
`conv_dw_int8_runtime` (0/4096 vs Teflon), `mm_nt_det GRID=1`. The `ROCKET_CONV_I8_FDBANK_EXTRA`
sentinel is kept (relative to the `+1` base) for future RE.
