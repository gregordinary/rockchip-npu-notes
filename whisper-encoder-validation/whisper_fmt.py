#!/usr/bin/env python3
# Parse whisper.cpp's legacy ggml model format (magic 'ggml', NOT GGUF).
# Returns {name: np.ndarray} with ggml ne-order shape (ne[0] fastest/contiguous),
# so a 2D linear weight comes back as shape (out, in) row-major == nn.Linear.
import struct, sys, numpy as np

GGML_MAGIC = 0x67676d6c  # 'ggml'
# ggml type -> (numpy dtype, bytes/elem). base.en uses only F32(0) and F16(1).
GTYPE = {0: (np.float32, 4), 1: (np.float16, 2)}

def _r_i32(f): return struct.unpack('<i', f.read(4))[0]
def _r_u32(f): return struct.unpack('<I', f.read(4))[0]

def load(path):
    f = open(path, 'rb')
    magic = _r_u32(f)
    assert magic == GGML_MAGIC, f"bad magic {magic:#x} (not whisper-ggml)"
    hp = {}
    for k in ['n_vocab','n_audio_ctx','n_audio_state','n_audio_head','n_audio_layer',
              'n_text_ctx','n_text_state','n_text_head','n_text_layer','n_mels','ftype']:
        hp[k] = _r_i32(f)
    # mel filters
    n_mel = _r_i32(f); n_fft = _r_i32(f)
    f.read(n_mel * n_fft * 4)  # f32 filter data
    # vocab
    n_vocab = _r_i32(f)
    for _ in range(n_vocab):
        ln = _r_u32(f)
        f.read(ln)
    # tensors
    tensors = {}
    while True:
        hdr = f.read(12)
        if len(hdr) < 12:
            break
        n_dims, length, ttype = struct.unpack('<iii', hdr)
        ne = [_r_i32(f) for _ in range(n_dims)]
        name = f.read(length).decode('utf-8', 'replace')
        dt, bpe = GTYPE[ttype]
        nelem = 1
        for e in ne: nelem *= e
        data = np.frombuffer(f.read(nelem * bpe), dtype=dt)
        # ne is fastest-first; reverse to numpy row-major shape (out, in) for 2D
        shape = tuple(reversed(ne))
        tensors[name] = data.reshape(shape)
    f.close()
    return hp, tensors

if __name__ == '__main__':
    hp, t = load(sys.argv[1])
    print("hparams:", {k: hp[k] for k in ['n_audio_ctx','n_audio_state','n_audio_head','n_audio_layer','n_mels','ftype']})
    print(f"{len(t)} tensors total")
    print("=== encoder layer 0 + front/post (name shape dtype) ===")
    for n in sorted(t):
        if 'decoder' in n: continue
        keep = (n.startswith('encoder.') and ('.blocks.0.' in n or 'blocks' not in n))
        if keep:
            print(f"  {n:42s} {tuple(t[n].shape)} {t[n].dtype}")
