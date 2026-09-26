#!/bin/bash
# DOES PINNING INTERACT WITH A KNOB THAT IS NOT A RESIDENCY KNOB? The third cell of the 2x2.
#
# WHAT IT PRICES. Every ratio in perf/data/tuning-matrix.md takes both arms UNPINNED, which is
# sound only if pinning does not interact with the knob under test. Two knobs are now measured
# and both interact NEGATIVELY: gemma4-12b F16 residency, 0.9854 against a 6.1 pp knob, and
# qwen35-9b quant residency, 0.9646 against a 66.8 pp one. Neither the interaction factor nor the
# absolute pin-gain loss is constant between them, so neither predicts a third by arithmetic.
#
# BOTH MEASURED KNOBS ARE RESIDENCY KNOBS, and the mechanism offered for them -- residency has
# already removed the A76 host pack work that pinning was accelerating -- predicts a NULL on a
# knob that touches neither. `-b 2048 -ub 2048` alone is that knob: it amortizes the
# per-micro-batch dequant and places nothing. On this model it is a resolved 1.424x (per-pass
# 1.408 1.428 1.437), so it is large enough to carry an interaction and is NOT a residency knob.
#
#     interaction = (ub_pin / stock_pin) / (ub_unpin / stock_unpin)
#
# THE MODEL IS HELD FIXED AGAINST THE SECOND MEASURED KNOB. qwen35-9b Q4_K_M carries both this
# knob and the quant-residency one, so the pair differ in the KNOB and not in the unit -- which is
# what lets a difference between their interactions be read as being about residency rather than
# about the model.
#
# NOT A PERF LEVER, SO NO CAP APPLIES. Nothing here proposes to make anything faster; the output
# is the trustworthiness of a column that already ships.
#
# WHY SIX PASSES. Both prior interaction contrasts needed them, and a pass budget is a property of
# the CONTRAST: this same model at this same shape reads 1.3% per-pass paired-ratio sd on one
# contrast where the 12B F16 unit reads 0.6-0.9% on another. The effect sought is a few pp.
#
# THE PINNED ARMS ARE PIN_MASK=0xf0 WITH `-t 4`, matching both earlier 2x2s exactly, so the three
# interaction factors are comparable. A COMMAND WRAPPER CANNOT RIDE IN THE ENV FIELD -- bench-llm.sh
# builds `env $envs ROCKET_LOG_STDERR=1 $BIN ...`, so a `taskset` there eats the assignment after
# it and the arm dies rc=127 in 46 seconds while still writing a full plausible <!--RO--> line. The
# mask travels as PIN_MASK and BIN points at the wrapper for EVERY arm, pinned or not.
#
# READ THE ENGAGEMENT LINE PER ARM. No arm here should place anything: stock and ub2048 both run
# without a residency knob. An arm that reports resident weights is not this contrast and says so.
#
# CHECK THE <!--DATA--> ROW COUNT AFTER PASS 1: 4 arms, so 4 rows.
set -u
OUTD=${OUTD:-$PWD/trackd20-pinub}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/qwen35/Qwen3.5-9B-Q4_K_M.gguf"
export PINWRAP_BIN=${PINWRAP_BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export BIN=${BIN:-$MODELS/llama-bench-pinwrap.sh}
export TESTS=${TESTS:-"-p 2048 -n 0 -r 3"}
export PASSES=${PASSES:-6}
export OUT="$OUTD/qwen35-9b-ub-pin2x2.md"
export ERRD="$OUTD/err"
export ARMS='stock_unpin||
stock_pin|PIN_MASK=0xf0|-t 4
ub_unpin||-b 2048 -ub 2048
ub_pin|PIN_MASK=0xf0|-b 2048 -ub 2048 -t 4'
LOG="$OUTD/driver.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "NOT EXECUTABLE: $BIN" >&2; exit 2; }
[ -x "$PINWRAP_BIN" ] || { echo "MISSING: $PINWRAP_BIN" >&2; exit 2; }
[ -f "$SO" ]    || { echo "MISSING: $SO" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- qwen35-9b Q4_K_M, pinning x -b 2048 -ub 2048 (a NON-residency knob) 2x2"
  echo "     TESTS='$TESTS'  PASSES=$PASSES"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     stock_unpin | stock_pin = PIN_MASK=0xf0 -t 4"
  echo "     ub_unpin    = -b 2048 -ub 2048   (the published 1.424x arm)"
  echo "     ub_pin      = the same + PIN_MASK=0xf0 -t 4"
  echo "     interaction = (ub_pin/stock_pin) / (ub_unpin/stock_unpin), paired within a pass"
  echo "     no arm places resident weights; an arm that reports any is not this contrast"
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 4 arms, RO on, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" qwen35-9b-ub-pin2x2
  rc=$?
  echo "[$(date -Is)] rc=$rc ALL-DONE"
} >> "$LOG" 2>&1
