#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 The rockchip-npu-notes authors
# Build one-op ONNX Sigmoid and Tanh models and compile each to an rk3588 .rknn with
# rknn-toolkit2 (fp16, no quantization). The .rknn embeds the vendor's NPU command
# stream, including the DPU LUT program and its table upload; decode_lut_rknn.py reads
# them back out. Compilation is offline and needs no NPU.
#
# Usage:  <venv>/bin/python gen_lut_rknn.py [outdir]
import os
import sys

import onnx
from onnx import helper, TensorProto

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# C=8 is one fp16 C2 plane; C=16 at 32x32 is two planes and a wider cube.
SHAPES = {"c8": [1, 8, 4, 4], "c16": [1, 16, 32, 32]}
OPS = ["Sigmoid", "Tanh"]


def make(op, shape, name):
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)
    node = helper.make_node(op, ["input"], ["output"])
    graph = helper.make_graph([node], name, [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path = os.path.join(OUT, name + ".onnx")
    onnx.save(model, path)
    return path


def convert(onnx_path, rknn_path):
    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk3588")
    ok = (rknn.load_onnx(model=onnx_path) == 0
          and rknn.build(do_quantization=False) == 0
          and rknn.export_rknn(rknn_path) == 0)
    rknn.release()
    return ok


if __name__ == "__main__":
    for op in OPS:
        for sname, shape in SHAPES.items():
            name = f"{op.lower()}_{sname}_f16"
            ok = convert(make(op, shape, name), os.path.join(OUT, name + ".rknn"))
            print(name, "OK" if ok else "FAILED")
