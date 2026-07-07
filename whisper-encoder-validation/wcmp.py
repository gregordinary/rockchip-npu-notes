#!/usr/bin/env python3
# Phase 1.2 compare: per-block cosine + max_abs of the NPU rocket encoder block
# (rout_iso / rout_chn, fp16) vs the f64 whisper-exact golden (gout).
import sys, os, numpy as np
DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/wenc"
T, d, nh, dff, nl, eps = open(os.path.join(DIR, "manifest.txt")).read().split()
T, d, nl = int(T), int(d), int(nl)

def cos(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0: return 1.0
    return float(a @ b / (na * nb)) if na and nb else 0.0

def ld32(n): return np.fromfile(os.path.join(DIR, n), dtype=np.float32).reshape(T, d).astype(np.float64)
def ld16(n): return np.fromfile(os.path.join(DIR, n), dtype=np.float16).reshape(T, d).astype(np.float64)

print(f"per-block fidelity  (T={T} d={d} nl={nl})")
print(f"{'blk':>3} | {'iso cos':>10} {'iso maxabs':>11} | {'chn cos':>10} {'chn maxabs':>11} | {'gold rms':>9}")
for L in range(nl):
    g = ld32(f"gout{L}.bin")
    riso = ld16(f"rout_iso{L}.bin")
    rchn = ld16(f"rout_chn{L}.bin")
    ci, mi = cos(g, riso), float(np.abs(g - riso).max())
    cc, mc = cos(g, rchn), float(np.abs(g - rchn).max())
    print(f"{L:>3} | {ci:>10.6f} {mi:>11.5f} | {cc:>10.6f} {mc:>11.5f} | {np.sqrt((g**2).mean()):>9.4f}")
print("\nisolated = block fed golden input (per-block error)")
print("chained  = block fed previous NPU output (compounding end-to-end fidelity)")
