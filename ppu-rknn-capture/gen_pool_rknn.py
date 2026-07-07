#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Build tiny 1-op ONNX pooling models and convert each to a .rknn for rk3588 with
# rknn-toolkit2 (the gahingwoo capture method, SOURCES.md). The .rknn embeds the
# vendor's NPU command stream; decode_pool_rknn.py walks it for the 64-bit regcmd
# words and decodes the PPU (0x6000) / PPU_RDMA (0x7000) register page.
#
# Purpose: a first-party ground-truth capture of the PPU pooling regcmd, used to
# establish the register program directly from hardware — specifically to crack the
# AVERAGE-pool RECIP_KERNEL fixed-point format (a magic-constant + host post-scale
# scheme). The avg kernels 2/3/4 give three distinct RECIP values to fit the
# encoding, which is then HW-validated (MAX bit-exact, AVG within fp16-recip tolerance).
#
# Runs offline on the x86 laptop venv (no NPU needed for conversion). Conversion is
# a compile; the rk3588-target regcmd is identical regardless of host arch.
#
# Usage:  /tmp/rknnvenv/bin/python gen_pool_rknn.py [outdir]
import os, sys
import onnx
from onnx import helper, TensorProto

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

def out_dim(i, k, s, p0, p1):
    return (i + p0 + p1 - k) // s + 1

def make_pool(name, op, C, H, W, kh=0, kw=0, sh=1, sw=1, pads=(0, 0, 0, 0), glob=False):
    """One pooling node, NCHW float input [1,C,H,W]."""
    inp = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, C, H, W])
    if glob:
        OH = OW = 1
        node = helper.make_node(op, ['input'], ['output'])
    else:
        OH = out_dim(H, kh, sh, pads[0], pads[2])
        OW = out_dim(W, kw, sw, pads[1], pads[3])
        node = helper.make_node(op, ['input'], ['output'],
                                kernel_shape=[kh, kw], strides=[sh, sw],
                                pads=[pads[0], pads[1], pads[2], pads[3]])
    out = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, C, OH, OW])
    graph = helper.make_graph([node], name, [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    p = os.path.join(OUT, name + ".onnx")
    onnx.save(model, p)
    print(f"  ONNX {name}: in 1x{C}x{H}x{W} -> out 1x{C}x{OH}x{OW}")
    return p

# Focused capture set. C=8 fills exactly one C2 plane (fp16 cube C2=8), the
# single-plane case; C=16 forces a 2-plane cube (surf-stride matters).
MODELS = [
    ("maxpool_2x2s2_c8",  "MaxPool",            8, 4, 4, 2, 2, 2, 2),  # structure + PC trailer, no recip
    ("avgpool_2x2s2_c8",  "AveragePool",        8, 4, 4, 2, 2, 2, 2),  # recip(k=2)
    ("avgpool_3x3s1_c8",  "AveragePool",        8, 6, 6, 3, 3, 1, 1),  # recip(k=3)
    ("avgpool_4x4s1_c8",  "AveragePool",        8, 4, 4, 4, 4, 1, 1),  # recip(k=4) (== global here)
    ("avgpool_2x3s1_c8",  "AveragePool",        8, 6, 6, 2, 3, 1, 1),  # asymmetric: recip_h(2) != recip_w(3)
    ("gavgpool_c16",      "GlobalAveragePool", 16, 4, 4),              # 2-plane + global avg
]

def convert(onnx_path, rknn_path):
    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    rknn.config(target_platform='rk3588')
    if rknn.load_onnx(model=onnx_path) != 0:
        print("    load_onnx FAILED"); rknn.release(); return False
    if rknn.build(do_quantization=False) != 0:
        print("    build FAILED"); rknn.release(); return False
    if rknn.export_rknn(rknn_path) != 0:
        print("    export FAILED"); rknn.release(); return False
    rknn.release()
    print(f"    -> {os.path.basename(rknn_path)} ({os.path.getsize(rknn_path)} bytes)")
    return True

if __name__ == "__main__":
    ok = 0
    for m in MODELS:
        name = m[0]
        print(f"[{name}]")
        kwargs = {}
        if m[1] == "GlobalAveragePool":
            op, C, H, W = m[1], m[2], m[3], m[4]
            onnx_p = make_pool(name, op, C, H, W, glob=True)
        else:
            op, C, H, W, kh, kw, sh, sw = m[1:9]
            onnx_p = make_pool(name, op, C, H, W, kh, kw, sh, sw)
        rknn_p = os.path.join(OUT, name + ".rknn")
        try:
            if convert(onnx_p, rknn_p):
                ok += 1
        except Exception as e:
            print(f"    convert EXC: {e}")
    print(f"\n{ok}/{len(MODELS)} converted")
