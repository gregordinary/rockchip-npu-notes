#!/usr/bin/env python3
"""Compare one FA op's dumped tiles between two runs, value by value.

ROCKET_FA_DUMP_OP=N writes op N's dense fp16 tiles and output as five raw files per run:
q [n_head][n_tokens][dk], k [n_kv_heads][n_kv][dk], v [n_kv_heads][dv][n_kv],
m [n_tokens][n_kv] and o [n_head][n_tokens][dv]. The shape comes from the run's
"ROCKET FA dump:" log line. For each tile this prints whether the two runs are byte-identical
and, where they are not, how many elements differ, by how many fp16 ulps, and where: per head
and per token for q and o, per KV head for k and v.

Usage: trackd40-fa-dump-diff.py DUMPDIR_A ERR_A DUMPDIR_B ERR_B
  DUMPDIR_*  a run's ROCKET_FA_DUMP_DIR
  ERR_*      that run's stderr, which carries the "ROCKET FA dump:" shape line
"""
import glob
import re
import sys

import numpy as np


def shape_of(err_path):
    pat = re.compile(r"ROCKET FA dump: op (\d+) n_tokens=(\d+) n_head=(\d+) n_kv=(\d+) "
                     r"n_kv_heads=(\d+) dk=(\d+) dv=(\d+)")
    with open(err_path, "rb") as f:
        for line in f:
            m = pat.search(line.decode("utf-8", "replace"))
            if m:
                op, nt, nh, nkv, nkvh, dk, dv = map(int, m.groups())
                return dict(op=op, nt=nt, nh=nh, nkv=nkv, nkvh=nkvh, dk=dk, dv=dv)
    sys.exit(f"no dump line in {err_path}")


def load(d, op, tag):
    paths = glob.glob(f"{d}/fa_op{op:04d}_pid*_{tag}.f16")
    if len(paths) != 1:
        sys.exit(f"{d}: expected one {tag} file for op {op}, found {len(paths)}")
    return np.fromfile(paths[0], dtype=np.float16)


def ulps(a, b):
    """Distance in fp16 units in the last place, sign-magnitude folded to a line."""
    ia = a.view(np.int16).astype(np.int32)
    ib = b.view(np.int16).astype(np.int32)
    ia = np.where(ia < 0, -32768 - ia, ia)
    ib = np.where(ib < 0, -32768 - ib, ib)
    return np.abs(ia - ib)


def report(tag, a, b, dims, names):
    if a.tobytes() == b.tobytes():
        print(f"  {tag}: byte-identical ({a.size} elements)")
        return
    a = a.reshape(dims)
    b = b.reshape(dims)
    u = ulps(a, b)
    nd = int((u > 0).sum())
    fin = np.isfinite(a.astype(np.float32)) & np.isfinite(b.astype(np.float32))
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    print(f"  {tag}: {nd} of {a.size} elements differ; ulps max {int(u.max())}, "
          f"count at 1..7 and 8+ ulps {np.bincount(np.minimum(u[u > 0], 8), minlength=9)[1:].tolist()}; "
          f"max |diff| {float(diff[fin].max()):.6g}; non-finite in either {int((~fin).sum())}")
    for ax, name in enumerate(names[:-1]):
        other = tuple(i for i in range(len(dims)) if i != ax)
        per = (u > 0).sum(axis=other)
        hit = np.nonzero(per)[0]
        print(f"    by {name}: {len(hit)} of {dims[ax]} carry a difference; "
              f"first {hit[:12].tolist()}{' ...' if len(hit) > 12 else ''}")


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    da, ea, db, eb = sys.argv[1:]
    sa, sb = shape_of(ea), shape_of(eb)
    if sa != sb:
        sys.exit(f"shapes differ: {sa} vs {sb}")
    s = sa
    print(f"op {s['op']}: n_tokens {s['nt']}, n_head {s['nh']}, n_kv {s['nkv']}, "
          f"n_kv_heads {s['nkvh']}, dk {s['dk']}, dv {s['dv']}")
    tiles = [
        ("q", (s["nh"], s["nt"], s["dk"]), ("head", "token", "dk")),
        ("k", (s["nkvh"], s["nkv"], s["dk"]), ("kv_head", "kv", "dk")),
        ("v", (s["nkvh"], s["dv"], s["nkv"]), ("kv_head", "dv", "kv")),
        ("m", (s["nt"], s["nkv"]), ("token", "kv")),
        ("o", (s["nh"], s["nt"], s["dv"]), ("head", "token", "dv")),
    ]
    for tag, dims, names in tiles:
        a, b = load(da, s["op"], tag), load(db, s["op"], tag)
        if a.size != b.size or a.size != int(np.prod(dims)):
            print(f"  {tag}: sizes {a.size} and {b.size}, expected {int(np.prod(dims))}")
            continue
        report(tag, a, b, dims, names)


if __name__ == "__main__":
    main()
