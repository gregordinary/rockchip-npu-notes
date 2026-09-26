#!/bin/bash
# WHAT ROCKET_N_THREADS COSTS AND BUYS ACROSS THE BINDER CROSSOVER, on the MoE route.
#
# WHAT IT PRICES. `ggml-rocket/API.md` prescribes raising ROCKET_N_THREADS as the exit from an
# exhausted NPU IOVA window. That advice has been priced only where the RAM floor bound
# (gemma4-12b F16: 0.9873x wall, 281 placed against 286). The pre-flight map says where the
# window can bind at all on this board: the pre-flight charges each stack `codes + scales + GGUF
# source` against RAM and `codes` alone against IOVA, and on gpt-oss-20b MXFP4 that ratio is
# 1.53826 (five recorded runs, budgets 11.7-25.7 GB, agreeing to 5 significant figures). IOVA
# binds first only where RAM_budget / (nw x 3840 MB) exceeds it, so at this board's ~24.5 GB
# default budget the window binds at nw <= 4 and the floor binds at nw >= 5.
#
# THE WINDOW QUESTION IS ALREADY SETTLED AND THIS DOES NOT RE-ASK IT. The pre-flight announced
# `Bound by RAM` in three runs at nw=5 and nw=8, with the admitted set BYTE-IDENTICAL across a
# window doubled from 19200 to 30720 MB -- 62 stacks, 24140 MB RAM, 15693 MB IOVA in all three
# [HW sweep 2026-09-02, RK1; raw in trackd17-moe-preflight]. So on this route the knob cannot buy
# placement, and what is left to price is what it COSTS.
#
# WHY THAT IS WORTH PASSES. `API.md` prescribes raising it here, and the only cost ever measured
# for the raise is 0.9873x on a DENSE F16 unit. The three -r 1 readings above read 28.36 / 28.20
# at nw=5 against 24.54 at nw=8 -- about ten times that, on the route the advice is written for.
# One -r 1 process is not a measurement and this is not treating it as one; it is why the cell is
# worth three rotated passes rather than being read off the recorded 12B number.
#
# THE ARMS. nt5 is the default, spelled as an ABSENCE so a default move is visible; nt8 is the
# prescribed raise, matching the 12B cell's definition exactly so the two units are comparable.
# nt6 is dropped for board time; it interpolates inside one regime.
#
# NO CAP APPLIES: this prices advice that already ships rather than proposing a lever.
#
# PASS BUDGET. Three to start, and the first pass is a pilot: this contrast has no recorded sd,
# and a pass budget is a property of the CONTRAST, not the unit. Extend from the measured
# per-pass paired-ratio sd, do not assume the 12B F16 unit's 0.6-0.9%.
set -u
OUTD=${OUTD:-$PWD/trackd17c-moe-ntvalue}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/gpt-oss-20b/gpt-oss-20b-mxfp4.gguf"
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export BIN=${BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export TESTS=${TESTS:-"-p 2048 -n 0 -r 3"}
export PASSES=${PASSES:-3}
export OUT="$OUTD/gptoss-20b-nthreads-ladder.md"
export ERRD="$OUTD/err"
export ARMS='nt5||-b 2048 -ub 2048
nt8|ROCKET_N_THREADS=8|-b 2048 -ub 2048'
LOG="$OUTD/driver.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "MISSING BIN: $BIN" >&2; exit 2; }
[ -f "$SO" ]    || { echo "MISSING SO: $SO" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- gpt-oss-20b MXFP4, ROCKET_N_THREADS ladder across the pre-flight binder crossover"
  echo "     TESTS='$TESTS'  PASSES=$PASSES  args='-b 2048 -ub 2048' on every arm"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     both arms are RAM-bound: the pre-flight announces Bound by RAM at 62 stacks either way"
  echo "     ratios are paired WITHIN a pass against nt5"
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 2 arms, RO on, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" gptoss-20b-nthreads-ladder
  rc=$?
  echo "[$(date -Is)] rc=$rc ALL-DONE"
} >> "$LOG" 2>&1
