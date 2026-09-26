#!/bin/bash
# TWO LOOSE ENDS FROM trackd34, BOTH OF WHICH ARE CONTROLS THAT WERE MISSING.
#
# ONE: IS THE PERPLEXITY DIFFERENCE REAL? trackd34 read 13.8994 with ROCKET_FLASH_ATTN=1 against
# 13.9024 with =0 and concluded the handler runs under llama-perplexity. That conclusion needs a
# control it did not have: perplexity is deterministic only if the whole stack is, and a parallel
# reduction whose order moves between processes would produce a difference of exactly this size
# with no handler involved. So the =1 arm is run TWICE. If the two =1 runs agree with each other
# and differ from =0, the handler ran and its summary was lost to the log. If the two =1 runs
# differ from each other by about as much, the instrument cannot see this and trackd34's reading
# was noise.
#
# TWO: DOES THE DRM_IOCTL_VERSION ROUTE PRINT? trackd34 asked for it at `-p 64` and got nothing,
# which says nothing: 64 rows may offload no matmul at all, so the device is never opened and the
# line has no occasion to print. Asked again at `-p 2048`, which is known to open the device on
# this model, and with ROCKET_LOG_LEVEL=debug beside ROCKET_DEBUG, since the tee is documented to
# apply only to a line that was emitted at all.
set -u
OUTD=${OUTD:-$PWD/trackd35-reach2}
MODEL=${MODEL:-${MODELS:?set MODELS to the directory holding the GGUFs}/ministral3-3b/Ministral-3-3B-Instruct-2512-F16.gguf}
CORPUS=${CORPUS:-$PWD/trackd33-fa-correctness/corpus.txt}
SO=$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so
PPL=$HOME/npu/llama.cpp/build/bin/llama-perplexity
BENCH=$HOME/npu/llama.cpp/build/bin/llama-bench
LOG="$OUTD/reach2.log"
mkdir -p "$OUTD"
[ -s "$CORPUS" ] || { echo "MISSING CORPUS" >&2; exit 2; }
{
  echo "[$(date -Is)] trackd35: the two controls trackd34 lacked"
  echo "  so=$(md5sum "$SO" | cut -d' ' -f1)"
  for arm in fa1a fa1b fa0; do
    case $arm in fa0) FA=0 ;; *) FA=1 ;; esac
    echo "--- $arm: ROCKET_FLASH_ATTN=$FA ---"
    sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    env GGML_BACKEND_PATH="$SO" ROCKET_KACC=1 ROCKET_FLASH_ATTN=$FA \
      taskset 0xf0 "$PPL" -m "$MODEL" -f "$CORPUS" -c 2048 --chunks 2 -t 4 -fa on \
      > "$OUTD/$arm.out" 2> "$OUTD/$arm.err"
    echo "  rc=$?"
    grep -ahoE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' "$OUTD/$arm.out" "$OUTD/$arm.err" | tail -1
    grep -ahoE '^\[[12]\][0-9.]+' "$OUTD/$arm.out" | tr '\n' ' '; echo
  done
  echo "--- driver version at -p 2048, debug level explicit ---"
  env GGML_BACKEND_PATH="$SO" ROCKET_DEBUG=1 ROCKET_LOG_LEVEL=debug ROCKET_LOG_STDERR=1 \
    taskset 0xf0 "$BENCH" -m "$MODEL" -p 2048 -n 0 -r 1 -t 4 -v \
    > "$OUTD/ver.out" 2> "$OUTD/ver.err"
  echo "  rc=$?"
  grep -a "opened rocket" "$OUTD/ver.err" | head -1 || echo "  (still no 'opened rocket' line)"
  echo "  rocket lines seen: $(grep -ac 'rocket' "$OUTD/ver.err")"
  echo "[$(date -Is)] TRACKD35-DONE"
} >> "$LOG" 2>&1
