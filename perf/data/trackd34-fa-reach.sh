#!/bin/bash
# DOES llama-perplexity REACH THE FA HANDLER, OR IS ITS SUMMARY BEING DROPPED?
#
# THE OBSERVATION. llama-perplexity prints neither `ROCKET FA total` nor `ROCKET FA checksum`,
# where llama-bench on the same model, `.so` and prompt length prints both. Two causes look
# identical from outside: the scheduler placed no FLASH_ATTN_EXT node on this backend under that
# tool's graph, or the node ran and the line was dropped -- both summaries come from an `atexit`
# handler, and llama.cpp's `common_log`, which `common_init` installs and `llama-bench` does not,
# is asynchronous and discards messages once its worker is paused (`common/log.cpp`, `add()`).
#
# THE DISCRIMINATOR IS THE TOOL'S OWN PRINTED NUMBER, not another log line -- a second line
# emitted through the same channel would inherit the same doubt. The handler computes attention
# in fp16 through the NPU's QK/AV; the CPU backend computes it with its own tiled kernel. The two
# do not agree bit for bit, so a perplexity that MOVES between `ROCKET_FLASH_ATTN=1` and
# `ROCKET_FLASH_ATTN=0` says the handler ran and only its summary was lost. A perplexity
# identical to every printed digit says the handler never ran.
#
# The control is llama-bench under the same two settings: it prints the FA line, so it says the
# knob does what the arms assume, and it is what makes an identical perplexity readable as
# placement rather than as a knob that did nothing.
set -u
OUTD=${OUTD:-$PWD/trackd33-fa-reach}
MODEL=${MODEL:-${MODELS:?set MODELS to the directory holding the GGUFs}/ministral3-3b/Ministral-3-3B-Instruct-2512-F16.gguf}
CORPUS=${CORPUS:-$PWD/trackd33-fa-correctness/corpus.txt}
SO=$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so
PPL=$HOME/npu/llama.cpp/build/bin/llama-perplexity
BENCH=$HOME/npu/llama.cpp/build/bin/llama-bench
LOG="$OUTD/reach.log"
mkdir -p "$OUTD"
if [ ! -f "$CORPUS" ]; then
  CORPUS="$OUTD/corpus.txt"
  cat "$HOME/npu/ggml-rocket/API.md" "$HOME/npu/ggml-rocket/README.md" > "$CORPUS"
fi
[ -s "$CORPUS" ] || { echo "MISSING CORPUS: $CORPUS" >&2; exit 2; }
[ -x "$PPL" ] && [ -x "$BENCH" ] || { echo "MISSING BINARY" >&2; exit 2; }
{
  echo "[$(date -Is)] FA reach discriminator"
  echo "  model=$MODEL  so=$(md5sum "$SO" | cut -d' ' -f1)"
  for fa in 1 0; do
    echo "--- ROCKET_FLASH_ATTN=$fa, llama-perplexity ---"
    sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    env GGML_BACKEND_PATH="$SO" ROCKET_KACC=1 ROCKET_FLASH_ATTN=$fa \
      ROCKET_FA_TIMING=1 ROCKET_FA_CHECKSUM=1 \
      taskset 0xf0 "$PPL" -m "$MODEL" -f "$CORPUS" -c 2048 --chunks 2 -t 4 -fa on \
      > "$OUTD/ppl-fa$fa.out" 2> "$OUTD/ppl-fa$fa.err"
    echo "  rc=$?"
    grep -ahoE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' "$OUTD/ppl-fa$fa.out" "$OUTD/ppl-fa$fa.err" | tail -1
    grep -a "ROCKET FA" "$OUTD/ppl-fa$fa.err" | tail -2 || echo "  (no FA line, as expected for this tool)"
  done
  for fa in 1 0; do
    echo "--- ROCKET_FLASH_ATTN=$fa, llama-bench CONTROL ---"
    env GGML_BACKEND_PATH="$SO" ROCKET_KACC=1 ROCKET_FLASH_ATTN=$fa \
      ROCKET_FA_TIMING=1 ROCKET_FA_CHECKSUM=1 \
      taskset 0xf0 "$BENCH" -m "$MODEL" -p 2048 -n 0 -r 1 -t 4 -v \
      > "$OUTD/bench-fa$fa.out" 2> "$OUTD/bench-fa$fa.err"
    echo "  rc=$?"
    grep -a "ROCKET FA" "$OUTD/bench-fa$fa.err" | tail -2 || echo "  (no FA line: the knob is off, which is the point)"
  done
  # A second question, answered for free by a process that opens the device: can the rocket DRM
  # driver's version be read after the dmesg line has scrolled away? The standing advice is that
  # `dmesg | grep "Initialized rocket"` is the only way to tell 1.6.0 from 1.5.0, and that matters
  # because dpu_grace_us=0 means opposite things on the two. The ring buffer rolls: on the RK3576,
  # up 6 days, the line is gone and the module declares no MODULE_VERSION. The library already
  # reads the version through DRM_IOCTL_VERSION at every open and prints it under ROCKET_DEBUG,
  # which is a route that does not depend on the ring buffer.
  echo "--- driver version through DRM_IOCTL_VERSION, not dmesg ---"
  env GGML_BACKEND_PATH="$SO" ROCKET_DEBUG=1 ROCKET_LOG_STDERR=1 \
    "$BENCH" -m "$MODEL" -p 64 -n 0 -r 1 -t 4 2>&1 | grep -a "opened rocket" | head -1 \
    || echo "  (no 'opened rocket' line -- the route does not work as read from the source)"
  echo "--- what dmesg still has, for comparison ---"
  sudo -n dmesg 2>/dev/null | grep -a "Initialized rocket" | tail -1 \
    || echo "  (dmesg no longer carries it on this board)"
  echo "[$(date -Is)] FA-REACH-DONE"
} >> "$LOG" 2>&1
