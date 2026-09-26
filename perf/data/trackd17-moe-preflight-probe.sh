#!/bin/bash
# WHICH LIMIT DOES THE MoE PRE-FLIGHT NAME? The gate for pricing ROCKET_N_THREADS where the
# advice applies.
#
# `ggml-rocket/API.md` prescribes raising ROCKET_N_THREADS as the exit from an exhausted NPU
# IOVA window. That advice has only ever been priced on a unit where the RAM floor bound
# (gemma4-12b F16: 0.9873x wall, 281 placed against 286). The pre-flight prints WHICH budget
# bound it -- `Bound by NPU IOVA` or `Bound by RAM` -- and the two have opposite exits, so the
# reason string is the gate, not the ratio.
#
# NOT A DATA POINT, AN INSTRUMENT READING. This probe answers three things before any campaign
# is budgeted:
#   1. Which limit each MoE model's pre-flight names at the campaign shape (-b 2048 -ub 2048,
#      -p 2048), and whether the string is STABLE across repeats -- a first-decline reason is a
#      race when the two limits coincide.
#   2. Whether ROCKET_N_THREADS 5->8 moves placement on that model at all, and in which
#      direction. On a RAM-bound unit more fds LOWER placement; a gain is evidence the window
#      was binding.
#   3. The per-arm wall cost, so the campaign's pass budget comes from a measured arm.
#
# -r 1, deliberately: the placement lines and the reason string are properties of the process,
# not of the timed reps, and this probe is not timing anything.
set -u
OUTD=${OUTD:-$PWD/trackd17-moe-preflight}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
BIN=${BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
mkdir -p "$OUTD"
LOG="$OUTD/probe.log"
: > "$LOG"

[ -f "$SO" ] || { echo "MISSING SO: $SO" >&2; exit 2; }
[ -x "$BIN" ] || { echo "MISSING BIN: $BIN" >&2; exit 2; }

reset_mem() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1
              echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null 2>&1; }

# label | model path | ROCKET_N_THREADS ('' = default 5)
UNITS='gptoss_nt5_a|gpt-oss-20b/gpt-oss-20b-mxfp4.gguf|
gptoss_nt5_b|gpt-oss-20b/gpt-oss-20b-mxfp4.gguf|
gptoss_nt8|gpt-oss-20b/gpt-oss-20b-mxfp4.gguf|8
qwen330_nt5|qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf|
qwen330_nt8|qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf|8'

echo "[$(date -Is)] probe start  so=$(md5sum "$SO" | cut -d' ' -f1)" >> "$LOG"
printf '%s\n' "$UNITS" | while IFS='|' read -r label rel nt; do
  [ -n "$label" ] || continue
  M="$MODELS/$rel"
  [ -f "$M" ] || { echo "MISSING MODEL: $M" >> "$LOG"; continue; }
  ERR="$OUTD/$label.err"
  reset_mem
  ma=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  t0=$(date +%s)
  echo "[$(date -Is)] === $label  nt='${nt:-default}'  MemAvail=${ma}kB" >> "$LOG"
  env GGML_BACKEND_PATH="$SO" ROCKET_KACC=1 ROCKET_LOG_STDERR=1 \
      ${nt:+ROCKET_N_THREADS=$nt} \
      "$BIN" -m "$M" -p 2048 -n 0 -r 1 -b 2048 -ub 2048 -o md \
      > "$OUTD/$label.md" 2> "$ERR"
  rc=$?
  t1=$(date +%s)
  echo "[$(date -Is)] $label rc=$rc wall=$((t1-t0))s" >> "$LOG"
  echo "--- moe/resident lines ---" >> "$LOG"
  grep -E "moe-int8|f16-resident|IOVA|Bound by|resident budget|NPU IOVA window" "$ERR" >> "$LOG" 2>/dev/null
  echo "--- result ---" >> "$LOG"
  grep -E "pp2048" "$OUTD/$label.md" >> "$LOG" 2>/dev/null
  echo >> "$LOG"
done
echo "[$(date -Is)] ALL-DONE" >> "$LOG"
