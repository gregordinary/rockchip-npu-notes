# CBUF bank slack for the int8 feature DMA

A matmul or convolution tile declares how many of the 12 CBUF banks its input feature
occupies and how many its weight occupies. The `DATA_BANK` and `WEIGHT_BANK` fields of
`CNA_CBUF_CON0` (`0x1040`) hold the two counts [source-confirmed: Mesa `registers.xml`], and
the two partition the 12 banks. The exact fit is `data_bank = ceil(feature_bytes / 32 KB)`,
just enough banks to hold the cube. For the int8 feature cube, the exact fit is one bank too
few (Erratum E4) [HW sweep].

## Programming requirements

**Conditions:** RK3588, an int8 or uint8 feature cube, matmul or convolution.

Software must reserve one slack bank for the int8 feature [HW sweep]:

- `data_bank = min(fd_banks + 1, 11)`, where `fd_banks` is the exact fit.
- `weight_bank = 12 − data_bank`.

The weight always has room: its per-kernel bytes are <= one bank, and it never needs all the
leftover. A tiler must reserve the matching bank, so that it never picks a tile whose feature
cannot get its `+1`. Software must budget feature and weight to 11 banks, not 12
(`I8_BUDGET = CBUF_BANKS − 1`). The cost is zero for real model shapes, since `Kt` is
unchanged. Only feature tiles already near 11 banks shrink by one K-step.

> [!NOTE]
> The failing tile geometries have no closed form, so no arithmetic test predicts them. Give
> every int8 feature the slack bank.

The fp16 feature cube does not resonate, and keeps the exact-fit sizing [HW sweep]. An fp16
control at the identical `(Mtile=144, Ktile=672)` tile is bit-exact while int8 corrupts.

## Software support

The matmul generator is `gen_matmul_int8`, run as a 1×1 convolution. The host tiler budgets
feature and weight to 11 banks (`I8_BUDGET`).

The int8 and uint8 convolution path has the same exact-fit sizing and the same fix. Its
generator is `gen_conv2d_int8_fill`, shared by `gen_conv2d_int8` and `gen_conv2d_dw_int8`. Its
direct-convolution tiler is `conv2d_int8_run`. Without the slack bank, both use
`data_bank = ceil(feat_bytes/bank)`, and the tiler budgets feature and weight to all 12 banks.
The fix mirrors the matmul:

- `gen_conv2d_int8_fill`: `data_bank = min(fd_banks + 1, 11)`, `weight_bank = 12 −
  data_bank` (shared by direct and depthwise).
- `conv2d_int8_run` (direct tiler): reserve the matching bank,
  `feat_budget = (12 − 1 − weight_banks)·32 KB`, int8 feature ceiling `CONV_FEAT_BUDGET_I8
  = 7` banks (fp16 stays 8, it is immune).
- **Depthwise** (`conv2d_dw_int8_run`) needs no tiler change. Its weight is one per-channel
  `KH·KW·G`-byte cube (<= 1 bank) and its feature is capped at 8 banks. The shared
  generator's `+1` therefore always fits (`data_bank <= 9`, >= 3 banks left for the weight).

The `ROCKET_CONV_I8_FDBANK_EXTRA` sentinel is kept, relative to the `+1` base, for future
reverse-engineering.

## Known issues

### Erratum E4: garbage in the last rows of an int8 tile

- **Description:** The CNA feature-input DMA runs one bank past the exact-fit `ceil`
  allocation, and the last rows of the cube fall off the end. The failing geometries are a
  joint 2-D `(Mtile, Ktile)` resonance with no closed form: the failing K-groups depend on
  Mtile, and the reverse. With `feature_bytes = Mtile·Ktile`, they cluster where the cube
  nearly fills its last bank. A near-full bank is necessary and not sufficient: an exactly
  full bank is fine, and a 98%-full one corrupts.
- **Conditions:** RK3588 under the `rocket` driver. An int8 or uint8 feature cube with the
  exact-fit `DATA_BANK`. The task is a matmul at certain `(Mtile, Ktile)` tile geometries, or
  a convolution whose `datain_width` is 1. A convolution with `IW>=2` uses a different
  descriptor and does not resonate. The table below lists failing matmul tiles.
- **Symptom:** The last few output rows of the tile are garbage. Every column of those rows
  is wrong, with values ~1e5-1e6 against the correct ones, and everything else in the tile is
  bit-exact [HW sweep]. In a matmul that fans N across cores, the defect looks like a
  different defect class, nondeterminism over the worker count. The worker count changes the
  per-worker `Nt`, which changes `Kt`, which changes whether an emitted tile falls on a
  resonant `(Mtile, Ktile)`. The root cause is the same.
- **Workaround:** Software must program `DATA_BANK` as `min(fd_banks + 1, 11)` and
  `WEIGHT_BANK` as `12 − DATA_BANK` for an int8 or uint8 feature. Software must budget
  feature and weight to 11 banks.
- **Status:** Open.

Failing matmul tile geometries [HW sweep]:

| Mtile | failing K-groups (`K/32`) |
|---|---|
| 144 | 7, 21, 35 |
| 192 | 5, 21, 37 |
| 240 | 17, 21, 25, 29 |

## Evidence

### Programming requirements

A `ROCKET_I8_FDBANK_EXTRA` sentinel sweep on the resonant tile establishes the fix
[HW sweep]. `+1` is bit-exact, `−1` is far worse, and overriding
`surf_stride`/`feature_grains`/`data_entries` instead does not help. The fix is bit-exact
across a full `(M, K, N)` grid (86 failing cases -> 0) and end-to-end (EfficientDet nt=1 ≡
nt=4).

The fp16 feature cube is C2=8, 2-byte, and the int8 cube is C2=16, 1-byte (see
[tile-layouts.md](tile-layouts.md)). The descriptor stride and bank math (`line_stride`,
`surf_stride`, `data_entries`) is identical for both, sharing fp16's exact-fit bank sizing.
For the same `(M,K)`, the fp16 cube's 2x byte size places the last bank at a safe fill, so the
corruption is int8-cube-specific. The defect therefore stays latent on the heavily used fp16
path.

### Software support

The convolution test is `tflite-rocket/tests/conv_bank_slack.c`, the convolution analog of
`mm_nt_det.c`. The default `IW=1` hot-K sweep passes, and `ROCKET_CONV_I8_FDBANK_EXTRA=-1`
reverts to exact-fit and fails. These regressions are clean:

- `convert_test`
- `conv2d_int8_rocket`
- `conv2d_fp16_rocket`
- `conv_dw_int8_runtime`
- `mm_nt_det GRID=1`

### Known issues

The host int64 K-accumulation is exact, so the fault is the CNA feature-input DMA reading the
tail rows of the cube from the wrong place.

The convolution path is susceptible to the same resonance under two conditions. Its generator
uses the same exact-fit `data_bank`. Its direct-convolution tiler budgets feature and weight to
all 12 banks with no slack. Without the slack bank, the convolution path is safe by luck, not
by design.

With `datain_width = 1`, the convolution feature DMA descriptor matches the matmul's exactly
(`gen_matmul_int8` sets `datain_width=1, datain_height=M, datain_channel=K`). A single-job int8
1×1 convolution at `IW=1`, whose feature cube is the matmul's, shows the identical signature
[HW sweep]. At `IC=1184`, `IH=189..193` and `216..221` (97.6-99.8 % of the last bank), the last
output rows garble (~3e5-9e5) and all else is bit-exact. The output is clean at <=97 % fill,
the same sharp near-full threshold. `+1` clears the whole window (1494/1494 shapes bit-exact),
and `-1` worsens it. `IW>=2` shapes (a different descriptor) do not resonate, which is why real
detectors, all `IW≫1`, never hit it.
