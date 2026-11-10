# SPDX-License-Identifier: MIT
# Copyright (C) 2026 The rockchip-npu-notes authors
#
# The tiling ablation with eager attention: torch 2.4's opset-14 SDPA symbolic rejects the float
# scale Idefics3 passes, so export the same math as MatMul/Softmax. Everything else is export_all.
import os, sys
sys.path.insert(0, os.path.join(os.environ["SHARD_REPO"], "src"))
from smolvlm_convert import exporter

_orig = exporter.AutoModelForVision2Seq.from_pretrained


class _Eager:
    @staticmethod
    def from_pretrained(mid, **kw):
        return _orig(mid, attn_implementation="eager", **kw)


exporter.AutoModelForVision2Seq = _Eager
exporter.export_all(work_dir=os.path.abspath(sys.argv[1]), ablate_tiling=True)
