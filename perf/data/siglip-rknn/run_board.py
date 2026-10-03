# SPDX-License-Identifier: GPL-3.0-or-later
#
# Additional permission under GNU GPL version 3 section 7: if you modify this program, or any
# covered work, by linking or combining it with Rockchip's rknn-toolkit2 or rknn-toolkit-lite2
# (or a modified version of either), containing parts covered by the terms of Rockchip's license
# for them, the licensors of this program grant you additional permission to convey the
# resulting work.
#
# Board-side runner for the SHARD reproduction (rknn-toolkit-lite2 2.3.2, vendor rknpu).
#   run_board.py mono  <model.rknn> <data> <out> <mask 012|0>
#   run_board.py shard <pack_dir>   <data> <out>
# Timing follows SHARD's own scripts: time.time() around rknnlite inference, 5 warm-ups, 20 reps
# (sbc_run_monolithic.py), on the first eval image. Outputs are saved for host-side scoring.
import os, sys, glob, json, time
import numpy as np
from rknnlite.api import RKNNLite

mode, target, DATA, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
evals = sorted(glob.glob(f"{DATA}/eval/*/x0.npy"))
names = [os.path.basename(os.path.dirname(p)) for p in evals]
inputs = [np.load(p).astype(np.float32) for p in evals]


def timed(fn, x, warm=5, reps=20):
    for _ in range(warm):
        fn(x)
    t = []
    for _ in range(reps):
        a = time.time()
        fn(x)
        t.append(time.time() - a)
    t = np.array(t)
    return {"mean_s": float(t.mean()), "median_s": float(np.median(t)), "min_s": float(t.min()),
            "max_s": float(t.max()), "reps": reps, "warmups": warm}


res = {"mode": mode, "target": target, "images": names}
if mode == "mono":
    mask = {"012": RKNNLite.NPU_CORE_0_1_2, "0": RKNNLite.NPU_CORE_0}[sys.argv[5]]
    res["mask"] = sys.argv[5]
    rk = RKNNLite()
    assert rk.load_rknn(target) == 0
    assert rk.init_runtime(core_mask=mask) == 0
    run = lambda x: rk.inference(inputs=[x])[0]
    res["time"] = timed(run, inputs[0])
    for n, x in zip(names, inputs):
        np.save(f"{OUT}/{n}.out.npy", np.asarray(run(x), dtype=np.float32))
    rk.release()
else:
    sys.path.insert(0, os.environ["SHARD_SRC"])
    from smolvlm_infer.runner import RKNNBlockRunner  # SHARD's runner, unchanged
    blocks = []
    for i in range(12):  # HybridSplitEncoder.__init__: round-robin cores 0/1/2
        blocks.append(RKNNBlockRunner(f"{target}/l{i}_attn.rknn", core_id=i % 3))
        blocks.append(RKNNBlockRunner(f"{target}/l{i}_mlp.rknn", core_id=i % 3))

    def forward(x):  # HybridSplitEncoder.forward, without the torch wrapper
        x = x * 0.1
        for b in blocks:
            x = b.run(x)
        return x * 10.0

    res["time"] = timed(forward, inputs[0])
    per_block = []
    for i in range(24):  # one block's own wall, for attribution
        x = inputs[0] * 0.1
        per_block.append(timed(blocks[i].run, x, warm=2, reps=5)["median_s"])
    res["per_block_median_s"] = per_block
    for n, x in zip(names, inputs):
        np.save(f"{OUT}/{n}.out.npy", np.asarray(forward(x), dtype=np.float32))
        # teacher forcing (validate.py): each block fed the fp32 reference stream, x0.1 in, x10 out
        stream = np.load(f"{DATA}/eval/{n}/stream.npy")
        tf = np.stack([np.asarray(blocks[i].run(stream[i:i + 1] * 0.1), np.float32)[0] * 10.0
                       for i in range(24)])
        np.save(f"{OUT}/{n}.tf.npy", tf)
json.dump(res, open(f"{OUT}/result.json", "w"), indent=1)
print(json.dumps(res["time"]), res.get("per_block_median_s", "")[:4] if mode == "shard" else "")
