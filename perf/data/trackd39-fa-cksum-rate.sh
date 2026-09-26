#!/bin/bash
# THE RATE AT WHICH THE FA HANDLER DISAGREES WITH ITSELF.
#
# WHAT IS MEASURED. On one build and one config, 2 of 7 runs printed a different aggregate FA
# checksum (both at k=1), and then 6 runs under the per-op mode agreed on every one of 288 ops.
# Thirteen runs do not bound a rate near one in five, and every other FA question divides by it:
# an equality gate that compares one run to one run cannot be read until it is known.
#
# WHAT THIS ASKS. RUNS repeats at ONE setting, ROCKET_FA_THREADS=1, under ROCKET_FA_CHECKSUM=2.
# The modal aggregate hash is the reference, and every run's per-op lines are diffed against a
# run that printed it, so a disagreeing run names its ops as well as counting toward the rate.
#   * the same op index or shape in every disagreeing run -> a wrong branch on one geometry
#   * a different op each time -> a race or an uninitialized read in the worker path
#
# WHAT IT CANNOT SEE. How far a disagreeing surface moved, or which hash is correct: a hash
# moves on one bit as readily as on a surface. That wants a value dump.
#
# THE OP COUNT IS THE FIRST THING TO READ. 288 in every run so far; a run with a different count
# ran a different graph and its diff is not a diff of the same thing.
set -u
RUNS=${RUNS:-20}
OUTD=${OUTD:-$PWD/trackd39-cksum-rate}
MODEL=${MODEL:-${MODELS:?set MODELS to the directory holding the GGUFs}/gemma4/gemma-4-12b-it-F16.gguf}
BIN=$HOME/npu/llama.cpp/build/bin/llama-bench
SO=$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so
LOG="$OUTD/rate.log"
mkdir -p "$OUTD"
{
  echo "[$(date -Is)] FA per-op checksum, $RUNS runs at ROCKET_FA_THREADS=1"
  echo "  so=$(md5sum "$SO" | cut -d' ' -f1)"
  echo "  librocketnpu.a=$(md5sum /usr/local/lib/librocketnpu.a | cut -d' ' -f1)"
  for run in $(seq 1 "$RUNS"); do
    sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    env GGML_BACKEND_PATH="$SO" ROCKET_KACC=1 ROCKET_FA_THREADS=1 \
      ROCKET_FA_CHECKSUM=2 ROCKET_FA_TIMING=1 ROCKET_LOG_STDERR=1 \
      taskset 0xf0 "$BIN" -m "$MODEL" -p 2048 -n 0 -r 1 -t 4 -v \
      > "$OUTD/run$run.out" 2> "$OUTD/run$run.err"
    rc=$?
    h=$(grep -a "ROCKET FA checksum" "$OUTD/run$run.err" | tail -1 | awk '{print $4}')
    grep -a "ROCKET FA op " "$OUTD/run$run.err" | sed 's/.*ROCKET FA op /op /' > "$OUTD/ops$run.txt"
    echo "run $run rc=$rc hash=${h:-NONE} ops=$(wc -l < "$OUTD/ops$run.txt")"
  done
  mode=$(grep -h "^run " "$LOG" 2>/dev/null | awk '{print $4}' | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
  ref=$(grep -h "^run " "$LOG" | grep -m1 "$mode" | awk '{print $2}')
  echo "--- hash counts ---"
  grep -h "^run " "$LOG" | awk '{print $4}' | sort | uniq -c | sort -rn
  echo "--- per-op diffs against run $ref, which printed the modal ${mode#hash=} ---"
  for run in $(seq 1 "$RUNS"); do
    [ "$run" = "$ref" ] && continue
    n=$(diff "$OUTD/ops$ref.txt" "$OUTD/ops$run.txt" | grep -c '^<' || true)
    [ "$n" = 0 ] && continue
    echo "  run $ref vs run $run: $n differing op lines"
    diff "$OUTD/ops$ref.txt" "$OUTD/ops$run.txt" | head -12
  done
  echo "[$(date -Is)] RATE-DONE"
} >> "$LOG" 2>&1
