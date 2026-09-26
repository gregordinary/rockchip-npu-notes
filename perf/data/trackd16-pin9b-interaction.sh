#!/bin/bash
# DOES PINNING INTERACT WITH A LARGE PUBLISHED KNOB? The 2x2 on qwen35-9b's biggest matrix cell.
#
# WHAT IT PRICES. Every ratio in perf/data/tuning-matrix.md takes both arms UNPINNED, which is
# sound only if pinning does not interact with the knob under test. On gemma4-12b F16 it does:
# the interaction factor is 0.9854, 4.3 se below 1.00, same sign in three passes -- but that is
# -1.5pp against a 6.1pp knob, and two knobs at two sizes do not bound a third. This runs the
# largest resolved knob in the matrix: qwen35-9b's stock -> `+ROCKET_QUANT_RESIDENT=auto` at
# `-b 2048 -ub 2048`, published at 1.661x (1.639 1.661 1.681, 200 resident / 0 streamed).
#
#     interaction = (qres_pin / stock_pin) / (qres_unpin / stock_unpin)
#
# NOT A PERF LEVER, SO NO CAP APPLIES. Nothing here proposes to make anything faster; the output
# is the trustworthiness of a column that already ships.
#
# WHY SIX PASSES. This unit's per-pass paired-ratio sd is ~1.3% (1.639-1.681 over three passes),
# roughly twice the 12B F16 unit's 0.6-0.9%, and the effect sought is of the same order as the
# 12B's 1.5pp. Three passes would leave it unresolved. A pass budget is a property of the UNIT.
#
# THE PINNED ARMS ARE PIN_MASK=0xf0 WITH `-t 4`, matching the 12B 2x2's definition of "pinned"
# exactly, so the two units' interaction factors are comparable. A COMMAND WRAPPER CANNOT RIDE IN
# THE ENV FIELD -- bench-llm.sh builds `env $envs ROCKET_LOG_STDERR=1 $BIN ...`, so a `taskset`
# there eats the assignment after it and the arm dies rc=127 in 46 seconds while still writing a
# full plausible <!--RO--> line. The mask travels as PIN_MASK and BIN points at the wrapper for
# EVERY arm, pinned or not, so the wrapper is not itself a difference between arms.
#
# READ THE ENGAGEMENT LINE PER ARM. Both qres arms must report 200 resident / 0 streamed. A pair
# taken at different residency is not a ratio, and this model is the one that reports "0 streamed"
# -- if an arm does not, the pass is not comparable and says so rather than being averaged in.
#
# CHECK THE <!--DATA--> ROW COUNT AFTER PASS 1: 4 arms, so 4 rows. An arm that died rc=127 still
# writes a header, a <!--PRED--> line and a full <!--RO--> line.
set -u
OUTD=${OUTD:-$PWD/trackd16-pin9b}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/qwen35/Qwen3.5-9B-Q4_K_M.gguf"
export PINWRAP_BIN=${PINWRAP_BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export BIN=${BIN:-$MODELS/llama-bench-pinwrap.sh}
export TESTS=${TESTS:-"-p 2048 -n 0 -r 3"}
export PASSES=${PASSES:-6}
export OUT="$OUTD/qwen35-9b-qres-pin2x2.md"
export ERRD="$OUTD/err"
export ARMS='stock_unpin||
stock_pin|PIN_MASK=0xf0|-t 4
qres_unpin|ROCKET_QUANT_RESIDENT=auto|-b 2048 -ub 2048
qres_pin|ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0|-b 2048 -ub 2048 -t 4'
LOG="$OUTD/driver.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "NOT EXECUTABLE: $BIN" >&2; exit 2; }
[ -x "$PINWRAP_BIN" ] || { echo "MISSING: $PINWRAP_BIN" >&2; exit 2; }
[ -f "$SO" ]    || { echo "MISSING: $SO" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- qwen35-9b Q4_K_M, pinning x quant-residency 2x2  TESTS='$TESTS'  PASSES=$PASSES"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     stock_unpin | stock_pin  = PIN_MASK=0xf0 -t 4"
  echo "     qres_unpin  = ROCKET_QUANT_RESIDENT=auto -b 2048 -ub 2048   (the published 1.661x arm)"
  echo "     qres_pin    = the same + PIN_MASK=0xf0 -t 4"
  echo "     interaction = (qres_pin/stock_pin) / (qres_unpin/stock_unpin), paired within a pass"
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 4 arms, RO on, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" qwen35-9b-qres-pin2x2
  rc=$?
  echo "[$(date -Is)] rc=$rc ALL-DONE"
} >> "$LOG" 2>&1
