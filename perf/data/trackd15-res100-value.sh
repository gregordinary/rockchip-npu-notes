#!/bin/bash
# WHAT DOES FULL RESIDENCY BUY ON THE 12B -- and the control that keeps the answer readable.
#
# THE QUESTION. `ROCKET_F16_RESIDENT=auto` places 286 of 328 weights (18078MB, 87%) and prices at
# 1.0614x (se 0.0018) against the streamed configuration. `ROCKET_N_THREADS=8` together with
# `ROCKET_QUANT_RESIDENT_RESERVE_MB=6144` places 328 of 328 (20790MB, 0 streamed). Neither knob
# alone moves placement by one weight. What the last 42 weights are worth is open.
#
# WHY THREE ARMS AND NOT TWO. The 100%-resident configuration differs from the 87% one in TWO
# settings, and one of them is not a memory knob at all: ROCKET_N_THREADS is the NPU WORKER-FD
# COUNT (default 5, "hard knee at 5, 6+ is noise" on a pp1024/ub1024 sweep [source-confirmed,
# ggml-rocket.cpp:260-264] -- which is not this model at this shape). Raising it to 8 changes the
# worker split as well as the fd count, so a two-arm campaign would charge that to residency.
# `res87_t8` places the SAME 286 weights at the same threads as `res100`, so:
#     res100 / res87_t8  = what the last 42 weights buy, threads held fixed
#     res87_t8 / res87   = what the thread count buys, placement held fixed
#     res100 / res87     = what the whole recipe buys
#
# READ THE PLACED FRACTION ON EVERY ARM BEFORE READING ANY RATIO. res87 and res87_t8 must both
# report 286 of 328 and res100 must report 328 of 328. Placement varies run to run on a partly
# placed model, and an arm that placed differently is a different configuration, not a data point.
#
# THE OOM HAZARD, AND WHY IT IS BUDGETED RATHER THAN ASSUMED. res100 holds 2712MB more resident
# than res87 on a board with NO SWAP whose dmesg already carries three OOM kills. It was only ever
# run at `-p 512 -n 0 -r 1`, where it left MemAvailable at 6542MB. This campaign runs at -p 2048,
# whose KV cache is four times larger. trackd14b probes that headroom ONCE, unpaired, before this
# script is launched. A killed arm is void and an OOM-killed board costs the session.
set -u
OUTD=${OUTD:-$PWD/trackd15-res100}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/gemma4/gemma-4-12b-it-F16.gguf"
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export BIN=${BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export TESTS=${TESTS:-"-p 2048 -n 0 -r 3"}
export PASSES=${PASSES:-3}
export OUT="$OUTD/gemma4-12b-f16-res100.md"
export ERRD="$OUTD/err"
export ARMS='res87|ROCKET_F16_RESIDENT=auto|
res87_t8|ROCKET_F16_RESIDENT=auto ROCKET_N_THREADS=8|
res100|ROCKET_F16_RESIDENT=auto ROCKET_N_THREADS=8 ROCKET_QUANT_RESIDENT_RESERVE_MB=6144|'
LOG="$OUTD/driver.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "NOT EXECUTABLE: $BIN" >&2; exit 2; }
[ -f "$SO" ]    || { echo "MISSING: $SO" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- gemma4-12b F16, what full residency buys  TESTS='$TESTS'  PASSES=$PASSES"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     so=$(md5sum "$SO" | cut -d' ' -f1)"
  echo "     res87    = ROCKET_F16_RESIDENT=auto                        expect 286 of 328 (18078MB)"
  echo "     res87_t8 = + ROCKET_N_THREADS=8                            expect 286 of 328 (thread control)"
  echo "     res100   = + ROCKET_QUANT_RESIDENT_RESERVE_MB=6144         expect 328 of 328 (20790MB, 0 streamed)"
  echo "     All arms UNPINNED, matching every other ratio in the matrix. Pinning interacts with"
  echo "     the residency knob on this model by -1.5pp against a 6.1pp knob, so a ratio of this"
  echo "     size is read unpinned and the interaction is a separate row."
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 3 arms, RO on, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" gemma4-12b-f16-res100
  rc=$?
  echo "[$(date -Is)] rc=$rc ALL-DONE"
} >> "$LOG" 2>&1
