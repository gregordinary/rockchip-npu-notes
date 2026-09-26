#!/bin/bash
# THE FA CHECKSUM GATE CAME BACK RED, AND THE FIRST QUESTION IS WHETHER IT REPRODUCES.
# k=1 read 345a87c827fa972b where k=4 read 8852206bf3ccda62, and k=4 is the value the
# previous build recorded for BOTH arms. Two candidates look identical from one run each:
# the k=1 arm is now deterministically different (a build-attributable change), or the
# handler is nondeterministic at k=1 and the previous session's single pair agreed by luck.
# Four runs separate them: k=1 three times and k=4 twice, interleaved so a drift cannot
# sit on one arm.
set -u
OUTD=${OUTD:-$PWD/trackd37-cksum-repro}
MODEL=${MODEL:-${MODELS:?set MODELS to the directory holding the GGUFs}/gemma4/gemma-4-12b-it-F16.gguf}
BIN=$HOME/npu/llama.cpp/build/bin/llama-bench
SO=$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so
LOG="$OUTD/repro.log"
mkdir -p "$OUTD"
{
  echo "[$(date -Is)] FA checksum reproduction"
  echo "  so=$(md5sum "$SO" | cut -d' ' -f1)"
  echo "  librocketnpu.a=$(md5sum /usr/local/lib/librocketnpu.a | cut -d' ' -f1)"
  echo "  recorded on the previous build: k=1 and k=4 both 8852206bf3ccda62, 288 ops, 2818572288 bytes"
  for run in 1 2 3 4 5; do
    case $run in 1|3|5) k=1 ;; *) k=4 ;; esac
    echo "--- run $run: ROCKET_FA_THREADS=$k ---"
    sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    env GGML_BACKEND_PATH="$SO" ROCKET_KACC=1 ROCKET_FA_THREADS=$k \
      ROCKET_FA_CHECKSUM=1 ROCKET_FA_TIMING=1 ROCKET_LOG_STDERR=1 \
      taskset 0xf0 "$BIN" -m "$MODEL" -p 2048 -n 0 -r 1 -t 4 -v \
      > "$OUTD/run$run-k$k.out" 2> "$OUTD/run$run-k$k.err"
    echo "  rc=$?"
    grep -a "ROCKET FA checksum" "$OUTD/run$run-k$k.err" | tail -1 || echo "  NO CHECKSUM LINE"
    grep -a "ROCKET FA total" "$OUTD/run$run-k$k.err" | tail -1
  done
  echo "[$(date -Is)] REPRO-DONE"
} >> "$LOG" 2>&1
