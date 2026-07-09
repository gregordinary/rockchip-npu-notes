#!/usr/bin/env python3
# Phase 1.2 prep: load REAL whisper encoder weights, compute a double-precision
# whisper-EXACT golden encoder (tanh-GELU, biased-var LN eps=1e-5, attn scale
# 1/sqrt(d_head), K has no bias), and export weights + per-block golden in/out as
# flat binaries for the C NPU harness (wvalidate.c).
#
# Weights are exported as fp16 (the dtype rocket consumes); both golden and rocket
# therefore see the IDENTICAL fp16-rounded weights, so any divergence is rocket's
# fp16/LUT path, not a weight mismatch. Golden math runs in f64.
import sys, os, struct, numpy as np
from whisper_fmt import load

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/wenc"
T   = int(sys.argv[2]) if len(sys.argv) > 2 else 256
MODEL = sys.argv[3] if len(sys.argv) > 3 else "whisper.cpp/models/ggml-base.en.bin"
X0F = sys.argv[4] if len(sys.argv) > 4 else None   # optional real encoder-input .bin (f32, T*d)
os.makedirs(OUT, exist_ok=True)
EPS = 1e-5

hp, t = load(MODEL)
d, nh, nl = hp['n_audio_state'], hp['n_audio_head'], hp['n_audio_layer']
dh = d // nh
dff = t['encoder.blocks.0.mlp.0.weight'].shape[0]   # (out,in) -> out = d_ff
print(f"model: d={d} nh={nh} nl={nl} dff={dff} n_ctx={hp['n_audio_ctx']}  T={T}")

def f16(a): return np.asarray(a, dtype=np.float16)
def save16(name, a): f16(a).tofile(os.path.join(OUT, name))
def save32(name, a): np.asarray(a, dtype=np.float32).tofile(os.path.join(OUT, name))

# fp16-rounded weights as f64 for the golden (matches what rocket gets)
def W(name): return f16(t[name]).astype(np.float64)

def layernorm(x, g, b):              # biased variance (ggml_norm), then affine
    m = x.mean(1, keepdims=True)
    v = ((x - m) ** 2).mean(1, keepdims=True)
    return (x - m) / np.sqrt(v + EPS) * g + b

def gelu_tanh(x):                    # ggml_gelu_f32 (tanh approx)
    c = 0.79788456080286535588; a = 0.044715
    return 0.5 * x * (1.0 + np.tanh(c * x * (1.0 + a * x * x)))

def attn(x, Wq, bq, Wk, Wv, bv, Wo, bo):
    Tt = x.shape[0]; scale = 1.0 / np.sqrt(dh)
    Q = x @ Wq.T + bq; K = x @ Wk.T; V = x @ Wv.T + bv     # (T,d); K no bias
    Q = Q.reshape(Tt, nh, dh).transpose(1, 0, 2)           # (nh,T,dh)
    K = K.reshape(Tt, nh, dh).transpose(1, 0, 2)
    V = V.reshape(Tt, nh, dh).transpose(1, 0, 2)
    s = (Q @ K.transpose(0, 2, 1)) * scale                 # (nh,T,T)
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s); p = e / e.sum(-1, keepdims=True)
    ctx = (p @ V).transpose(1, 0, 2).reshape(Tt, d)        # (T,d)
    return ctx @ Wo.T + bo

def block(x, L):
    p = f"encoder.blocks.{L}."
    ln = layernorm(x, W(p+"attn_ln.weight"), W(p+"attn_ln.bias"))
    a = attn(ln, W(p+"attn.query.weight"), W(p+"attn.query.bias"),
             W(p+"attn.key.weight"), W(p+"attn.value.weight"), W(p+"attn.value.bias"),
             W(p+"attn.out.weight"), W(p+"attn.out.bias"))
    xa = x + a
    ln2 = layernorm(xa, W(p+"mlp_ln.weight"), W(p+"mlp_ln.bias"))
    h = gelu_tanh(ln2 @ W(p+"mlp.0.weight").T + W(p+"mlp.0.bias"))
    f2 = h @ W(p+"mlp.2.weight").T + W(p+"mlp.2.bias")
    return xa + f2

# block-0 input: real captured (4th positional arg) or realistic synthetic (pos-embed + noise)
pe = t['encoder.positional_embedding'].astype(np.float64)
if X0F:
    x0 = np.fromfile(X0F, dtype=np.float32).astype(np.float64).reshape(T, d)
    print(f"x0: REAL captured from {X0F}  rms={np.sqrt((x0**2).mean()):.4f}")
else:
    rng = np.random.default_rng(1234)
    x0 = pe[:T] + 0.5 * rng.standard_normal((T, d))
    print(f"x0: SYNTHETIC (pos-embed[:T] + 0.5*randn)  rms={np.sqrt((x0**2).mean()):.4f}")
x0 = f16(x0).astype(np.float64)      # round to fp16 (what rocket gets)

# export weights (fp16) per layer + manifest
WK = [("ln1g","attn_ln.weight"),("ln1b","attn_ln.bias"),("wq","attn.query.weight"),
      ("bq","attn.query.bias"),("wk","attn.key.weight"),("wv","attn.value.weight"),
      ("bv","attn.value.bias"),("wo","attn.out.weight"),("bo","attn.out.bias"),
      ("ln2g","mlp_ln.weight"),("ln2b","mlp_ln.bias"),("wf1","mlp.0.weight"),
      ("bf1","mlp.0.bias"),("wf2","mlp.2.weight"),("bf2","mlp.2.bias")]
for L in range(nl):
    for short, full in WK:
        save16(f"w{L}_{short}.bin", t[f"encoder.blocks.{L}.{full}"])
save16("x0.bin", x0)
with open(os.path.join(OUT, "manifest.txt"), "w") as f:
    f.write(f"{T} {d} {nh} {dff} {nl} {EPS:.8e}\n")

# golden: chain blocks, save per-block input (gin) and output (gout) as f32
gin = x0.copy()
for L in range(nl):
    save32(f"gin{L}.bin", gin)
    gout = block(gin, L)
    save32(f"gout{L}.bin", gout)
    gin = gout
print(f"exported weights + golden for {nl} blocks to {OUT}")
print(f"final golden rms={np.sqrt((gin**2).mean()):.4f}")
