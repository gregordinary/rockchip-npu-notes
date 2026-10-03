# SPDX-License-Identifier: GPL-3.0-or-later
# Real-image encoder inputs and the fp32 reference for the SHARD reproduction.
# Each image is resized to exactly 512x512 so all 1024 patches are valid (no attention mask).
# Saves, per image: x0 = encoder input [1,1024,768] (post patch+pos embedding), the residual
# stream after every attention and MLP block (24 tensors), the encoder output and post-LN output.
import os, sys, json
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
OUT = sys.argv[1]
EVAL = sys.argv[2].split(",")
CALIB = sys.argv[3].split(",") if len(sys.argv) > 3 else []
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
proc = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()
vm = model.model.vision_model

if os.environ.get("POS_FIX") == "1":
    # transformers 4.55-4.57 build the position coordinates as k/nb*(1-1e-6), which on a full
    # 32x32 grid falls just below each bucket boundary and shifts every position id down one
    # (row/col 0 used twice, 31 never). 4.46-4.53 and 5.0 map a full grid to the identity.
    # Every image here is a full 512x512 grid, so the identity is the model as trained.
    E = vm.embeddings

    def _forward(pixel_values, patch_attention_mask):
        assert bool(patch_attention_mask.all())
        x = E.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        return x + E.position_embedding.weight[None]

    E.forward = _forward
    print("POS_FIX: identity position ids")
print("attn impl:", model.config._attn_implementation, "| vision config:",
      vm.config.hidden_size, vm.config.num_hidden_layers, vm.config.image_size, vm.config.patch_size)


def embed(path):
    img = Image.open(path).convert("RGB").resize((512, 512), Image.BICUBIC)
    ip = proc.image_processor(images=[[img]], do_image_splitting=False, return_tensors="pt")
    pv = ip["pixel_values"][0]  # [n_img, 3, H, W]
    assert pv.shape == (1, 3, 512, 512), pv.shape
    if "pixel_attention_mask" in ip:
        assert bool(ip["pixel_attention_mask"].all()), "padding present"
    pam = torch.ones(1, 32, 32, dtype=torch.bool)
    with torch.no_grad():
        x0 = vm.embeddings(pixel_values=pv, patch_attention_mask=pam)
    return pv, x0


meta = {}
for tag, paths in (("eval", EVAL), ("calib", CALIB)):
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        pv, x0 = embed(p)
        d = os.path.join(OUT, tag, name)
        os.makedirs(d, exist_ok=True)
        np.save(f"{d}/pixel.npy", pv.numpy().astype(np.float32))
        np.save(f"{d}/x0.npy", x0.numpy().astype(np.float32))
        if tag == "eval":
            stream = [x0.numpy()]
            h = x0
            with torch.no_grad():
                for L in vm.encoder.layers:
                    a = L.self_attn(L.layer_norm1(h))
                    h = h + (a[0] if isinstance(a, tuple) else a)
                    stream.append(h.numpy())
                    h = h + L.mlp(L.layer_norm2(h))
                    stream.append(h.numpy())
                enc = vm.encoder(inputs_embeds=x0).last_hidden_state
                post = vm.post_layernorm(enc)
            # the hand-run stream and the module's own forward must agree
            dev = float((torch.from_numpy(stream[-1]) - enc).abs().max())
            np.save(f"{d}/stream.npy", np.concatenate(stream, 0).astype(np.float32))  # [25,1024,768]
            np.save(f"{d}/enc.npy", enc.numpy().astype(np.float32))
            np.save(f"{d}/post.npy", post.numpy().astype(np.float32))
            meta[name] = {"stream_vs_forward_maxabs": dev, "x0_std": float(x0.std()),
                          "enc_absmax": float(enc.abs().max()), "enc_std": float(enc.std()),
                          "stream_absmax": [float(np.abs(s).max()) for s in stream]}
            print(name, json.dumps({k: v for k, v in meta[name].items() if k != "stream_absmax"}),
                  "absmax by block:", [round(v, 1) for v in meta[name]["stream_absmax"]])
json.dump(meta, open(os.path.join(OUT, "ref_meta.json"), "w"), indent=1)
