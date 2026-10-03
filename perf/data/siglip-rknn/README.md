# Raw data for the SigLIP-B/16 RKNN and SHARD comparison

Behind [../siglip-encoder.md](../siglip-encoder.md). Measured 2026-10-03 on two Turing RK1
modules: one on `6.1.172-vendor-rk35xx` with `rknpu` 0.9.8, one on mainline `7.2.8-1-arm64`
with `rocket` 1.3.0.

The models come from SmolVLM-256M-Instruct, built by rknn-toolkit2 2.3.2 with SHARD's own
scripts from `github.com/poad42/smolvlm_rk3588_full_npu_native` at `fe3eefc`. The board runs
them through `rknn-toolkit-lite2` 2.3.2 over `librknnrt` 2.3.2, which `rknnlite` loads only from
`/usr/lib`. Neither SHARD's code nor Rockchip's packages are redistributed here. The scripts
import SHARD's modules from a checkout named by `SHARD_REPO` or `SHARD_SRC`.

| file | what |
|---|---|
| `prep.py` | Real-image inputs and the fp32 reference: per image, the encoder input, the residual stream after all 24 blocks, the encoder output and the post-LN output |
| `build.py` | The RKNN models, each with the SHARD script its comment names: fp16 whole encoder, int8 on noise calibration, int8 on real calibration, the SHARD pack |
| `build_notile.py` | The SHARD pack built without tiling, with eager attention in the export |
| `run_board.py` | The board runner: SHARD's timing method, the outputs, and the teacher-forced block outputs for a pack |
| `score.py` | Encoder-output, post-LN and teacher-forced cosines against the reference, with the identity floor beside the last |
| `rknn-runs.txt` | The RKNN and SHARD timings and scores |
| `rocket-runs.txt` | The `siglip_rocket` gate on the same four images, resident bench included |
| `generate.txt` | SHARD's `run_ablation_tiling.py` on the board: three generates pinned, two on `ondemand`, and the caption |
| `perf-detail-fp16mono-012.txt` | The runtime's per-op profile of the fp16 whole encoder on cores 0-2, perf collection on |
| `transpose-warnings.txt` | The toolkit's fallback warnings for one layer built with and without SHARD's `score_transpose` probe |

Three of SHARD's scripts need care at `fe3eefc`:

- `export_vanilla_baseline.py` and `run_inference.py` do not parse. A `\n` inside several string literals is a literal line break. Restoring the escapes is the only change the build log here needed.
- `run_ablation_tiling.py` adds `../src` to its path from `scripts/experiments/`, which names no directory. Run it from `scripts/`.
- The opset-14 exports need eager attention under torch 2.4. Its ONNX exporter rejects the float `scale` Idefics3 passes to `scaled_dot_product_attention`. The math is unchanged.

## Commands

On an x86 host, Python 3.12, with `SHARD` naming the checkout. The second `prep.py` call builds
the int8 calibration set, without `POS_FIX`:

```sh
uv venv -p 3.12 venv && VIRTUAL_ENV=$PWD/venv uv pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  "rknn-toolkit2==2.3.2" "torch==2.4.0+cpu" "transformers==4.57.1" "numpy==1.26.4" \
  "onnx==1.16.1" onnxruntime pillow "setuptools<70"
POS_FIX=1 venv/bin/python prep.py data2 "$SHARD/data/vlm_example.jpg,<three COCO images>"
venv/bin/python prep.py data "$SHARD/data/vlm_example.jpg" "<eight COCO images>"
SHARD_REPO=$SHARD venv/bin/python build.py work data fp16mono int8broken int8calib
SHARD_REPO=$SHARD venv/bin/python build.py work2 data shard
SHARD_REPO=$SHARD venv/bin/python build_notile.py work2/shard_notile
```

On the board, Python 3.11 with `rknn-toolkit-lite2==2.3.2` and `numpy`, `librknnrt.so` in
`/usr/lib`, and the CPU and DDR governors set to `performance`:

```sh
venv/bin/python run_board.py mono models/fp16mono.rknn data2 out2/fp16mono_012 012
SHARD_SRC=$SHARD/src venv/bin/python run_board.py shard models/shard_pack data2 out2/shard_pack
venv/bin/python score.py data2 out2/fp16mono_012 out2/shard_pack
```

`setuptools<70` supplies the `pkg_resources` module rknn-toolkit2 imports. The eval images are
`000000000139`, `000000000285` and `000000000632`. The calibration images are `000000020247`,
`000000020333`, `000000020553`, `000000020571`, `000000020992`, `000000021167`, `000000021465`
and `000000021503`.
