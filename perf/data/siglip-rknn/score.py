# SPDX-License-Identifier: GPL-3.0-or-later
# Score board outputs against the fp32 reference.
#   score.py <data> <outdir>...
# End-to-end: cosine of the encoder output (all 1024x768 values, fp64) against fp32 PyTorch.
# Teacher-forced (shard packs): per-block cosine, SHARD's metric, beside the IDENTITY floor:
# what a block that returned its input unchanged would score, cos(stream[i], stream[i+1]).
import os, sys, json, glob
import numpy as np

DATA = sys.argv[1]


def cos(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


P = np.load(f"{DATA}/post_ln.npz")


def post_ln(x):  # the vision model's post_layernorm, fp64 on the host, applied to both sides
    x = x.astype(np.float64)
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + float(P["eps"])) * P["w"] + P["b"]


for out in sys.argv[2:]:
    res = json.load(open(f"{out}/result.json"))
    print(f"== {out}  time {res['time']}")
    for n in res["images"]:
        ref = np.load(f"{DATA}/eval/{n}/enc.npy")
        got = np.load(f"{out}/{n}.out.npy").reshape(ref.shape)
        finite = bool(np.isfinite(got).all())
        line = (f"  {n:<16} e2e cos {cos(ref, got):.6f}  post-LN cos {cos(post_ln(ref), post_ln(got)):.6f}"
                f"  maxabs err {np.abs(ref - got).max():.3g}  finite {finite}")
        tfp = f"{out}/{n}.tf.npy"
        if os.path.exists(tfp):
            s = np.load(f"{DATA}/eval/{n}/stream.npy")
            tf = np.load(tfp)
            c = [cos(s[i + 1], tf[i]) for i in range(24)]
            floor = [cos(s[i + 1], s[i]) for i in range(24)]
            line += (f"\n    teacher-forced per block: mean {np.mean(c):.4f} min {np.min(c):.4f}"
                     f" (#<0.99: {sum(x < 0.99 for x in c)})"
                     f"\n    identity floor per block: mean {np.mean(floor):.4f} min {np.min(floor):.4f}"
                     f"\n    blocks: " + " ".join(f"{x:.4f}" for x in c))
        print(line)
