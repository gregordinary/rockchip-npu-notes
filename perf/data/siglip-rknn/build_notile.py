# SPDX-License-Identifier: GPL-3.0-or-later
#
# Additional permission under GNU GPL version 3 section 7: if you modify this program, or any
# covered work, by linking or combining it with Rockchip's rknn-toolkit2 or rknn-toolkit-lite2
# (or a modified version of either), containing parts covered by the terms of Rockchip's license
# for them, the licensors of this program grant you additional permission to convey the
# resulting work.
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
