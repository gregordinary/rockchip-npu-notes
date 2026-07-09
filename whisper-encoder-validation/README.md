# Whisper encoder on the NPU — validation + in-model integration recipe

Reproduction scripts for the Whisper-encoder-on-the-FOSS-NPU work written up in
[encodings/whisper-encoder.md](../encodings/whisper-encoder.md) §"Validation against whisper.cpp"
and §"In-model fused integration". Two things are reproduced here:

1. **Per-layer fidelity** of `rocket_encoder_block_fp16` vs whisper.cpp's real `base.en` encoder
   (result: per-block cos 1.000000, final post-LN cos **0.99981** vs whisper's `embd_enc` on real
   audio; a double-precision golden of whisper's exact math is cross-checked == whisper at 0.99983).
2. **In-model integration**: the whole encoder running on the NPU in stock whisper.cpp, producing a
   **byte-identical transcript** at base.en / small.en / medium.en (but 3.4–4.4× slower — the fused
   path is transfer-bound; see the note for why this is a correctness milestone, not a perf win).

## Files

| File | Role |
|---|---|
| `whisper_fmt.py` | Parser for whisper.cpp's legacy ggml model format (magic `ggml`, **not** GGUF). Returns `{name: ndarray}` in `(out,in)` row-major (== `nn.Linear`, == rocket's weight layout). |
| `gelu_block_diff.c` | Standalone (no NPU): quantifies whisper's tanh-GELU vs rocket's erf-GELU at the block level. Result: cos 1.0, max_abs 1.7e-4 → GELU variant is a non-issue. |
| `wprep.py` | Loads real encoder weights, computes a double-precision **whisper-exact golden** (biased-var LayerNorm eps=1e-5, tanh-GELU, attn scale 1/√dh, K no bias), exports weights (fp16) + per-block golden in/out for the C harness. Optional 4th positional arg: real captured encoder input (`.bin`). |
| `wvalidate.c` | C harness: runs `rocket_encoder_block_fp16` on the NPU per block (isolated + chained), writes outputs. Build: `gcc -O2 wvalidate.c -I <rocket-userspace>/include -I/usr/include/libdrm -L <build> -lrocketnpu -lpthread -lm -ldrm`. |
| `wcmp.py` | Per-block cosine / max_abs of NPU vs golden. |
| `apply_dump_patch.py` | Idempotent whisper.cpp patch (env `WHISPER_DUMP_ENC=1`) dumping the real encoder input (`embd_conv`) + output (`embd_enc`). Temporary instrumentation; revert with the `.bak`. |
| `wreal_prep.py` | Builds the real encoder input `inpL = pos_embed + transpose(embd_conv)` from the dump + stashes whisper's real `embd_enc`. |
| `wfinal.py` | Applies whisper's post-LN to golden + NPU outputs, compares both to the real `embd_enc` (the decisive golden==whisper and rocket==whisper cross-check). |
| `apply_rocket_enc_patch.py` | Idempotent whisper.cpp + CMake patch wiring the fused on-NPU encoder: each layer → `ggml_map_custom1` → `rocket_encoder_block_fp16`. Build gate `-DWHISPER_ROCKET=ON`, run gate `WHISPER_ROCKET_ENC=1` (both default off → stock build/drop-in unaffected). |

## Recipe (on an RK3588 with a built rocket-userspace + whisper.cpp)

```bash
# fidelity, synthetic-realistic input (real weights), T=1500
python3 wprep.py /tmp/wenc 1500 /path/to/ggml-base.en.bin
gcc -O2 wvalidate.c -I <ru>/include -I/usr/include/libdrm -L <ru>/build -lrocketnpu -lpthread -lm -ldrm -o wvalidate
./wvalidate /tmp/wenc && python3 wcmp.py /tmp/wenc

# fidelity vs whisper's REAL activations: dump, build inpL, run, cross-check
python3 apply_dump_patch.py /path/to/whisper.cpp   # then rebuild whisper-cli
WHISPER_DUMP_ENC=1 whisper-cli -m ggml-base.en.bin -f samples/jfk.wav   # writes /tmp/whisper_dump/
python3 wreal_prep.py
python3 wprep.py /tmp/wenc_real 1500 /path/to/ggml-base.en.bin /tmp/inpL_real.bin
./wvalidate /tmp/wenc_real && python3 wfinal.py /tmp/wenc_real

# in-model fused encoder (byte-identical transcript)
python3 apply_rocket_enc_patch.py /path/to/whisper.cpp
cmake -S whisper.cpp -B whisper.cpp/build -DWHISPER_ROCKET=ON -DROCKETNPU_DIR=/abs/path/to/rocket-userspace && cmake --build whisper.cpp/build -j8 --target whisper-cli
#   (ROCKETNPU_DIR defaults to a sibling ../../rocket-userspace; it must be built into its build/ dir first)
WHISPER_ROCKET_ENC=1 [ROCKET_ATTN_HOST_SOFTMAX=1] whisper-cli -m ggml-base.en.bin -f samples/jfk.wav
```
