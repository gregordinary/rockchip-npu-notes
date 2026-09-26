#!/bin/bash
# A SECOND INTERACTION CELL ON gemma4-12b F16, WITH A KNOB THAT IS NOT A HOST-WORK KNOB.
#
# WHAT IS BEING TESTED. Three interaction cells exist and a form appears in them: the pin-gain
# loss is a fixed FRACTION of the knob's size within one model, 0.0583 for both `qwen35-9b` knobs
# (2.41 pp on 41.4, 3.89 pp on 66.8, agreeing to four decimals) against 0.239 for the single
# `gemma4-12b` F16 cell [perf/data/tuning-matrix.md]. Two points do not establish a
# proportionality and one point establishes nothing at all, so the 12B constant is the weaker of
# the two and a second knob on that model is what tests it.
#
# WHY NOT THE THIRD qwen35-9b KNOB THE PLAN NAMED. The arm cannot resolve on that model, and
# the arithmetic says so without buying a pass. Every knob left there acts on the matmul datapath,
# and that model's prefill is dequant-bound: `MM_ASYM` measures **+1.3%** on Qwen3.5-9B-Q4_K
# [perf/asymmetric-tile.md, HW sweep 2026-07-01] and is smaller still at the stock micro-batch,
# which is more dequant-bound than the ub2048 arm that number was taken on. At 1.3 pp the form
# predicts a deficit of 0.0583 x 1.3 = **0.08 pp**, against this model's 1.4% per-pass
# paired-ratio sd. Three standard errors would need on the order of a thousand passes. **The two
# knobs that ARE large on that model are large for the same reason -- both remove the
# per-micro-batch dequant -- so a third large knob of a different kind does not exist there.**
#
# WHY THIS CELL RESOLVES. `MM_ASYM` measures **+5.7%** on gemma4-12b F16 [same sweep, 14.22 ->
# 15.03 t/s at pp2048], and this unit's per-pass paired-ratio sd is **0.6-0.9%** against the 9B's
# 1.4%. The form predicts a deficit of 0.239 x 5.7 = **1.36 pp**, which is 3.7-5.8 se at six
# passes and 2.7-4.0 at three. So the pass budget is decided from the CONTRAST that will be read:
# three first, extended only if the measured spread asks for it.
#
# AND IT SEPARATES THE FORM FROM ITS MECHANISM. The mechanism recorded for the negative
# interaction is that pinning accelerates HOST work and every knob measured so far removes some
# of it -- the two residency knobs remove the A76 weight pack, the micro-batch knob removes the
# per-micro-batch dequant. `MM_ASYM` re-shapes the DEVICE tiling: it halves Nt so the CBUF fill
# does more MAC per pass, and the K-accumulation and the output volume are unchanged. That
# mechanism therefore predicts a NULL here, while the observed form predicts -1.4 pp. This is the
# first cell where the two disagree. **It is not a pure device knob** -- halving Nt doubles the
# N-tile count, so the per-tile host pack and de-tile calls do change -- so a non-null does not by
# itself refute the mechanism; it bounds how much of the interaction the host-work story can own.
#
#     interaction = (stock_pin / asym0_pin) / (stock_unpin / asym0_unpin)
#
# THE KNOB IS SPELLED AS THE DEFAULT, NOT AS A FLAG. `ROCKET_MM_ASYM` ships default-on
# [source-confirmed, `rocket_matmul.c`], so the baseline arm is the one that OPTS OUT and the knob
# arm is an absence. Naming them the other way would report a knob nobody can take.
#
# NOT A PERF LEVER, SO NO CAP APPLIES. The output is the trustworthiness of a column that ships.
#
# THE PINNED ARMS ARE PIN_MASK=0xf0 WITH `-t 4`, matching all three earlier interaction cells
# exactly, so the four factors are comparable. A COMMAND WRAPPER CANNOT RIDE IN THE ENV FIELD --
# bench-llm.sh builds `env $envs ROCKET_LOG_STDERR=1 $BIN ...`, so a `taskset` there eats the
# assignment after it and the arm dies rc=127 in 46 seconds while still writing a full plausible
# <!--RO--> line. The mask travels as PIN_MASK and BIN points at the wrapper for EVERY arm.
#
# READ THE ENGAGEMENT LINE PER ARM. No arm here takes a residency knob, and this model's F16
# residency default is a K gate it does not pass, so every arm should report zero placed weights.
# An arm that reports any is not this contrast and says so.
#
# CHECK THE <!--DATA--> ROW COUNT AFTER PASS 1: 4 arms, so 4 rows.
set -u
OUTD=${OUTD:-$PWD/trackd23-asympin12b}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/gemma4/gemma-4-12b-it-F16.gguf"
export PINWRAP_BIN=${PINWRAP_BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export BIN=${BIN:-$MODELS/llama-bench-pinwrap.sh}
export TESTS=${TESTS:-"-p 2048 -n 0 -r 3"}
export PASSES=${PASSES:-3}
export OUT="$OUTD/gemma4-12b-asym-pin2x2.md"
export ERRD="$OUTD/err"
export ARMS='asym0_unpin|ROCKET_MM_ASYM=0|
asym0_pin|ROCKET_MM_ASYM=0 PIN_MASK=0xf0|-t 4
stock_unpin||
stock_pin|PIN_MASK=0xf0|-t 4'
LOG="$OUTD/trackd23.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "NOT EXECUTABLE: $BIN" >&2; exit 2; }
[ -x "$PINWRAP_BIN" ] || { echo "MISSING: $PINWRAP_BIN" >&2; exit 2; }
[ -f "$SO" ]    || { echo "MISSING: $SO" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- gemma4-12b F16, pinning x ROCKET_MM_ASYM (a DEVICE-tiling knob) 2x2"
  echo "     TESTS='$TESTS'  PASSES=$PASSES"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     asym0_unpin = ROCKET_MM_ASYM=0, the knob DECLINED; it is the base of every ratio"
  echo "     stock_*     = the shipping default, ROCKET_MM_ASYM absent"
  echo "     *_pin       = PIN_MASK=0xf0 -t 4, matching all three earlier interaction cells"
  echo "     interaction = (stock_pin/asym0_pin) / (stock_unpin/asym0_unpin), paired within a pass"
  echo "     no arm places resident weights; an arm that reports any is not this contrast"
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 4 arms, RO on, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" gemma4-12b-asym-pin2x2
  rc=$?
  echo "[$(date -Is)] rc=$rc TRACKD23-ALL-DONE"
} >> "$LOG" 2>&1
