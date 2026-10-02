#!/bin/bash
# DO THE INPUTS OF THE OP THE FA HANDLER DISAGREES AT ALREADY DIFFER?
#
# WHAT IS KNOWN. On gemma4-12b F16 at pp2048 the handler's output takes one of two values, about
# one run in five (7 of 33), and every odd run first differs at op 189 (the timed pass's
# microbatch 2, layer 45); the other differing ops are op 189's dataflow (trackd39).
#
# WHAT THIS ASKS. RUNS repeats of trackd39's setting under ROCKET_FA_CHECKSUM=3, which adds the
# FNV-1a of the four dense fp16 tiles the driver is handed (q, k, v, mask) to every per-op line,
# and ROCKET_FA_DUMP_OP=189, which writes that op's tiles and fp16 output to raw files per run.
#   * op 189's q/k/v/m hashes equal across a modal and an odd run, its output hash not
#       -> the difference is made inside the handler or on the device
#   * an input hash differs -> it is made upstream, in an op that feeds layer 45
# The dumps give the magnitude: an odd run's output against a modal run's, element by element.
#
# WHAT IT CANNOT SEE. Which upstream op, if an input differs: the per-op lines cover only the
# offloaded attention. And the rate on this build is its own measurement: level 3 hashes about
# 22 MB an op more, so the timing it perturbs is not trackd39's.
#
# THE OP COUNT IS STILL THE FIRST THING TO READ: 288 in every run so far.
set -u
RUNS=${RUNS:-20}
DUMPOP=${DUMPOP:-189}
OUTD=${OUTD:-/path/to/data/trackd40-fa-inputs}
MODEL=${MODEL:-/path/to/data/gemma4/gemma-4-12b-it-F16.gguf}
BIN=${BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
SO=${SO:-$HOME/npu/ggml-rocket-fa3/build-dl/libggml-rocket.so}
LOG="$OUTD/inputs.log"
mkdir -p "$OUTD"
{
  echo "[$(date -Is)] FA per-op checksum level 3, $RUNS runs at ROCKET_FA_THREADS=1, dump op $DUMPOP"
  echo "  so=$(md5sum "$SO" | cut -d' ' -f1)"
  echo "  librocketnpu.a=$(md5sum /usr/local/lib/librocketnpu.a | cut -d' ' -f1)"
  echo "  governor=$(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor) power=$(cat /sys/bus/platform/drivers/rocket/fdab0000.npu/power/control)"
  for run in $(seq 1 "$RUNS"); do
    mkdir -p "$OUTD/dump$run"
    sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    env GGML_BACKEND_PATH="$SO" ROCKET_KACC=1 ROCKET_FA_THREADS=1 \
      ROCKET_FA_CHECKSUM=3 ROCKET_FA_DUMP_OP="$DUMPOP" ROCKET_FA_DUMP_DIR="$OUTD/dump$run" \
      ROCKET_FA_TIMING=1 ROCKET_LOG_STDERR=1 \
      taskset 0xf0 "$BIN" -m "$MODEL" -p 2048 -n 0 -r 1 -t 4 -v \
      > "$OUTD/run$run.out" 2> "$OUTD/run$run.err"
    rc=$?
    h=$(grep -a "ROCKET FA checksum" "$OUTD/run$run.err" | tail -1 | awk '{print $4}')
    grep -a "ROCKET FA op " "$OUTD/run$run.err" | sed 's/.*ROCKET FA op /op /' > "$OUTD/ops$run.txt"
    echo "run $run rc=$rc hash=${h:-NONE} ops=$(wc -l < "$OUTD/ops$run.txt") $(date -Is)"
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
    diff "$OUTD/ops$ref.txt" "$OUTD/ops$run.txt" | head -16
  done
  echo "[$(date -Is)] INPUTS-DONE"
} >> "$LOG" 2>&1
