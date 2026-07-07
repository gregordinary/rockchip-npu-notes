#!/usr/bin/env python3
# Build the REAL encoder block-0 input inpL = pe + transpose(embd_conv) from whisper's
# dump, and stash whisper's real final output embd_enc, both as f32 [T,d] for the
# golden/rocket cross-check.
import os, numpy as np
from whisper_fmt import load
MODEL = "whisper.cpp/models/ggml-base.en.bin"
DUMP = "/tmp/whisper_dump"
hp, t = load(MODEL)
d = hp['n_audio_state']; T = hp['n_audio_ctx']
# embd_conv ne=[1500,512] -> numpy (512,1500); transpose -> (T,d). embd_enc ne=[512,1500] -> (T,d).
conv = np.fromfile(f"{DUMP}/embd_conv.bin", np.float32).reshape(d, T).T          # (T,d)
pe   = t['encoder.positional_embedding'][:T].astype(np.float32)                  # (T,d)
inpL = (pe + conv).astype(np.float32)
enc  = np.fromfile(f"{DUMP}/embd_enc.bin",  np.float32).reshape(T, d)            # (T,d) post-LN
inpL.tofile("/tmp/inpL_real.bin")
enc.tofile("/tmp/embd_enc_ref.bin")
print(f"T={T} d={d}  inpL rms={np.sqrt((inpL**2).mean()):.4f}  embd_enc rms={np.sqrt((enc**2).mean()):.4f}")
print("wrote /tmp/inpL_real.bin (real encoder input) + /tmp/embd_enc_ref.bin (whisper final)")
