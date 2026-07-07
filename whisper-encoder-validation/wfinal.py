#!/usr/bin/env python3
# Final cross-check: apply whisper's post-LN (encoder.ln_post) to the golden and the
# rocket chained block-5 output, compare both to whisper's REAL embd_enc.
#   golden vs whisper  -> proves the numpy golden == whisper.cpp's encoder
#   rocket vs whisper  -> the headline: FOSS NPU encoder vs whisper, real audio
import sys, os, numpy as np
from whisper_fmt import load
DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/wenc_real"
MODEL = "whisper.cpp/models/ggml-base.en.bin"
EPS = 1e-5
hp, t = load(MODEL)
d = hp['n_audio_state']; T = hp['n_audio_ctx']; nl = hp['n_audio_layer']
eg = np.float16(t['encoder.ln_post.weight']).astype(np.float64)
eb = np.float16(t['encoder.ln_post.bias']).astype(np.float64)
def ln(x):
    m = x.mean(1, keepdims=True); v = ((x - m) ** 2).mean(1, keepdims=True)
    return (x - m) / np.sqrt(v + EPS) * eg + eb
def cos(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
enc = np.fromfile("/tmp/embd_enc_ref.bin", np.float32).reshape(T, d).astype(np.float64)
g5  = np.fromfile(f"{DIR}/gout{nl-1}.bin", np.float32).reshape(T, d).astype(np.float64)
r5  = np.fromfile(f"{DIR}/rout_chn{nl-1}.bin", np.float16).reshape(T, d).astype(np.float64)
gfin, rfin = ln(g5), ln(r5)
print(f"final encoder output (post-LN), T={T} d={d}:")
print(f"  golden  vs whisper embd_enc : cos={cos(gfin, enc):.8f}  maxabs={np.abs(gfin-enc).max():.5f}")
print(f"  ROCKET  vs whisper embd_enc : cos={cos(rfin, enc):.8f}  maxabs={np.abs(rfin-enc).max():.5f}")
print(f"  rocket  vs golden (final)   : cos={cos(rfin, gfin):.8f}  maxabs={np.abs(rfin-gfin).max():.5f}")
print(f"  (whisper embd_enc rms={np.sqrt((enc**2).mean()):.4f})")
