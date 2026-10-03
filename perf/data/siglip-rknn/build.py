# SPDX-License-Identifier: GPL-3.0-or-later
#
# Additional permission under GNU GPL version 3 section 7: if you modify this program, or any
# covered work, by linking or combining it with Rockchip's rknn-toolkit2 or rknn-toolkit-lite2
# (or a modified version of either), containing parts covered by the terms of Rockchip's license
# for them, the licensors of this program grant you additional permission to convey the
# resulting work.
#
# Build the RKNN models for the SHARD reproduction with rknn-toolkit2 2.3.2, each with SHARD's own
# recipe (the file named beside it). Verbose compile logs are kept: they carry the vendor wording.
#   build.py <work> <data> <which...>   which in: fp16mono int8broken int8calib shard shard_notile
import os, sys, glob, signal, subprocess
import numpy as np
import torch
from transformers import AutoModelForVision2Seq
from rknn.api import RKNN

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
WORK, DATA = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
WHICH = sys.argv[3:]
REPO = os.environ["SHARD_REPO"]
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)


def load(eager=False):
    kw = {"attn_implementation": "eager"} if eager else {}
    return AutoModelForVision2Seq.from_pretrained(MODEL_ID, **kw).eval()


def export_encoder(onnx_path, opset, eager):
    # export_partial_offload.py (opset 12) / export_monolithic.py (opset 14): the encoder alone,
    # a randn [1,1024,768] dummy, do_constant_folding.
    if os.path.exists(onnx_path):
        return
    enc = load(eager).model.vision_model.encoder
    torch.onnx.export(enc, (torch.randn(1, 1024, 768),), onnx_path, input_names=["in"],
                      output_names=["out"], opset_version=opset, do_constant_folding=True)


def build(name, onnx_path, cfg, quant, dataset=None):
    log = f"{WORK}/{name}.build.log"
    rk = RKNN(verbose=True, verbose_file=log)
    rk.config(target_platform="rk3588", **cfg)
    assert rk.load_onnx(model=onnx_path) == 0, "load_onnx"
    ret = rk.build(do_quantization=quant, dataset=dataset)
    print(name, "build ret", ret)
    if ret == 0:
        print(name, "export ret", rk.export_rknn(f"{WORK}/{name}.rknn"))
    rk.release()


def try_export(onnx_path, opset):
    # SHARD's scripts load the default attention; if the legacy exporter refuses SDPA at this
    # opset, fall back to eager attention (the same math) and say so.
    try:
        export_encoder(onnx_path, opset, eager=False)
        print(onnx_path, "exported with the default attention")
    except Exception as e:
        print(onnx_path, "default-attention export failed:", repr(e)[:300], "-> eager")
        export_encoder(onnx_path, opset, eager=True)


if "fp16mono" in WHICH:
    # export_partial_offload.py: opset 12, float_dtype fp16, default optimization_level, no quant.
    try_export("enc_op12.onnx", 12)
    build("fp16mono", "enc_op12.onnx", {"float_dtype": "float16"}, quant=False)

if "int8broken" in WHICH or "int8calib" in WHICH:
    # export_broken_int8.py reuses export_monolithic.py's opset-14 ONNX.
    try_export("enc_op14.onnx", 14)

if "int8broken" in WHICH:
    # export_broken_int8.py verbatim: 16 uniform(-1,1) samples with element [0,0,0] = 1000.0,
    # asymmetric_quantized-8, optimization_level 3.
    os.makedirs("calib_broken", exist_ok=True)
    rng = np.random.default_rng(0)
    paths = []
    for i in range(16):
        d = rng.uniform(-1.0, 1.0, size=(1, 1024, 768)).astype(np.float32)
        d[0, 0, 0] = 1000.0
        p = os.path.abspath(f"calib_broken/calib_{i}.npy")
        np.save(p, d)
        paths.append(p)
    open("ds_broken.txt", "w").write("\n".join(paths) + "\n")
    build("int8broken", "enc_op14.onnx",
          {"quantized_dtype": "asymmetric_quantized-8", "optimization_level": 3},
          quant=True, dataset="ds_broken.txt")

if "int8calib" in WHICH:
    # The same config; only the calibration differs: real-image encoder inputs.
    paths = sorted(glob.glob(f"{DATA}/calib/*/x0.npy"))
    assert paths, "no calibration embeddings"
    open("ds_calib.txt", "w").write("\n".join(paths) + "\n")
    build("int8calib", "enc_op14.onnx",
          {"quantized_dtype": "asymmetric_quantized-8", "optimization_level": 3},
          quant=True, dataset="ds_calib.txt")

if "shard" in WHICH or "shard_notile" in WHICH:
    sys.path.insert(0, os.path.join(REPO, "src"))
    from smolvlm_convert import exporter
    if "shard" in WHICH:
        exporter.export_all(work_dir=f"{WORK}/shard_pack")
    if "shard_notile" in WHICH:
        exporter.export_all(work_dir=f"{WORK}/shard_notile", ablate_tiling=True)
