#!/bin/bash
# THE QUANT-RESIDENCY INGEST'S CPU COST, AND THE UNCONTAMINATED HOST SHARE ON THE 9B.
#
# WHAT IS SETTLED. `H/t = a/phi` returns the host term's share of prefill wall. On `qwen35-9b`
# Q4_K_M two knobs give 0.669 (quant residency) and 0.591 (`-b 2048 -ub 2048`), agreeing to 13%,
# with `g` -- pinning's speedup of host work -- agreeing to 1% [tuning-matrix.md]. `wall_s` and
# both `busy_tot` snapshots bracket the WHOLE `llama-bench` invocation, so a residency arm pays
# its one-time ingest inside that bracket and its `phi` reads LOW, which makes `a/phi` read HIGH.
# The residency arm carries a 37.7 s non-timed window, the micro-batch arm 3.6 s against a 2.5 s
# base, so the biased row is the one reading high.
#
# THE TWO KNOBS ARE NESTED, AND THE WRITE-UP CALLS THEM INDEPENDENT. The residency arm IS
# `ROCKET_QUANT_RESIDENT=auto` ON TOP OF `-b 2048 -ub 2048` [trackd16 header]. They share the
# micro-batch component, so a device-side term in `-ub` inflates BOTH rows and their agreement
# cannot see it. What the check actually is: a two-point collinearity test through the origin in
# (`phi`, `a`), with no residual degree of freedom. The "incremental knob" reading is
# algebraically the same equation, not a third point.
#
# THE ARM. Not a 2x2. `busy_tot = intercept + (r+1)*slope`, because llama-bench runs one internal
# warm-up plus `r` reps. Regressing `busy_tot` on the rep count separates them:
#
#   slope      = per-rep host work        -> phi = 1 - slope_res/slope_stock, uncontaminated
#   intercept  = model load + the ingest  -> the ingest itself, as core-seconds
#
# Six arms, PINNED only (`PIN_MASK=0xf0 -t 4`; phi is defined at fixed pinning), true stock
# against `ROCKET_QUANT_RESIDENT=auto -b 2048 -ub 2048`, at `-r 1`, `-r 2`, `-r 4`.
#
# THE PREDICTION, IN THE UNIT THAT TRANSFERS [registered 2026-09-03]. The f16 route's ingest
# measured 72.0 core-seconds over 18078 MB = 3.98 ms PER RESIDENT MB. The quant route writes
# 13184 MB and adds a Q4_K decode over the same output bytes, 6.6-13.2 core-seconds on an A76. So
# the per-MB multiple is 1.13-1.25, the ingest is 59-66 core-seconds, phi = 0.664-0.671 and the
# host share is 0.596-0.603: the 13% gap closes to about 2%, NOT to zero. RIVALS, each a number:
# read-side dominated at the f16 route's 3.81 ms per SOURCE MB gives 20.6 core-s and phi = 0.622;
# the previous plan's "the ingest runs at the 12B's 0.73 cores" gives 25.2 core-s and
# phi = 0.627, and it anchors to a rate over a window whose length is not a property of the code;
# no contamination at all gives phi = 0.600. Exact agreement of the two rows needs a multiple of
# 1.40. PER-MB IS THE INVARIANT THAT TRANSFERS BETWEEN THE TWO ROUTES; CORES-OVER-A-WINDOW IS NOT.
#
# THE BUDGET, AGAINST THE SMALLEST ADJACENT GAP. The candidates are 0.600 / 0.622 / 0.627 /
# 0.664-0.671, so the tightest pair is 0.022 apart. The 12B's three-pass regression gave
# se(phi) = 0.018 at a slope ratio of 0.77; this unit's slope ratio is ~0.34, so the same relative
# slope precision predicts se ~0.008 and the candidates sit 1.1-3.6 se apart at the pessimistic
# end. PRINT THE PER-PASS phi AND READ THE se BEFORE MAKING ANY CLAIM. If it lands at the high
# end, buy passes deliberately rather than quoting a straddle.
#
# `a` MUST BE IMPORTED, AND THE IMPORT HAS A CHECK. This campaign is pinned-only, so it cannot
# produce the UNPINNED paired ratio `a` needs; that is trackd16's 1.660x. What this campaign CAN
# check is trackd16's PINNED ratio of 1.608x against its own -- a disagreement says the board
# state moved and the import is not licensed.
#
# NOT A PERF LEVER, SO NO CAP APPLIES. The output is a denominator.
#
# THE CONTROL THAT MUST SUCCEED. This model places 100% (200 weights, 13184 MB) where the 12B
# placed 87%, so the ingest here is a clean intercept with no per-call remainder. Placement has to
# be CONSTANT across the three rep counts or the regression is fitting three configurations:
# every res arm must report 200 / 13184 MB and every stock arm none. The teardown lines are
# grepped below.
#
# WHAT A GREEN RESULT WOULD NOT SHOW. It does not re-test the functional form -- five cells carry
# that -- and it does not fill the phi gap between 0.09 and 0.50, which is trackd28's cell.
#
# CHECK THE <!--DATA--> ROW COUNT AFTER PASS 1: 6 arms, so 6 rows.
set -u
OUTD=${OUTD:-$PWD/trackd29-busyint9b}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/qwen35/Qwen3.5-9B-Q4_K_M.gguf"
export PINWRAP_BIN=${PINWRAP_BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export BIN=${BIN:-$MODELS/llama-bench-pinwrap.sh}
# -r lives in each arm's args, so TESTS must NOT carry one.
export TESTS=${TESTS:-"-p 2048 -n 0"}
export PASSES=${PASSES:-3}
export OUT="$OUTD/qwen35-9b-busy-intercept.md"
export ERRD="$OUTD/err"
export ARMS='stock_r1|PIN_MASK=0xf0|-t 4 -r 1
stock_r2|PIN_MASK=0xf0|-t 4 -r 2
stock_r4|PIN_MASK=0xf0|-t 4 -r 4
qres_r1|ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0|-b 2048 -ub 2048 -t 4 -r 1
qres_r2|ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0|-b 2048 -ub 2048 -t 4 -r 2
qres_r4|ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0|-b 2048 -ub 2048 -t 4 -r 4'
LOG="$OUTD/trackd29.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "NOT EXECUTABLE: $BIN" >&2; exit 2; }
[ -x "$PINWRAP_BIN" ] || { echo "MISSING: $PINWRAP_BIN" >&2; exit 2; }
[ -f "$SO" ]    || { echo "MISSING: $SO" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- qwen35-9b Q4_K_M, busy_tot against rep count, PINNED arms only"
  echo "     TESTS='$TESTS'  PASSES=$PASSES  (-r lives in each arm's args)"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     stock_rN = true stock; qres_rN = ROCKET_QUANT_RESIDENT=auto -b 2048 -ub 2048"
  echo "     (that compound IS the published 1.661x arm and IS trackd16's qres cell)"
  echo "     all arms PIN_MASK=0xf0 -t 4; phi is defined at fixed pinning"
  echo "     busy_tot = intercept + (r+1)*slope; llama-bench runs one internal warm-up plus r reps"
  echo "     phi = 1 - slope_qres/slope_stock ; ingest = intercept_qres - intercept_stock"
  echo "     PREDICTION 2026-09-03c: per-MB multiple 1.13-1.25 of the f16 route's 3.98 ms/MB,"
  echo "       ingest 59-66 core-s over 13184 MB, phi 0.664-0.671, host share 0.596-0.603"
  echo "       RIVALS: phi 0.622 (read-side), 0.627 (0.73 cores), 0.600 (no contamination)"
  echo "       exact agreement of the two published rows would need a multiple of 1.40"
  echo "     IMPORT CHECK: trackd16's PINNED ratio is 1.608x; a disagreement voids the a import"
  echo "     CONTROL: every qres arm must place 200 weights / 13184 MB; every stock arm zero"
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 6 arms, RO on, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" qwen35-9b-busy-intercept
  rc=$?
  echo "[$(date -Is)] rc=$rc TRACKD29-ALL-DONE"
  echo "--- placement control: every qres arm 200/13184MB, every stock arm zero ---"
  for f in "$ERRD"/*.err; do
    printf '%s: ' "$(basename "$f")"
    grep -oE '\[(f16|quant)-resident\][^\n]*' "$f" | tail -1 || echo "(no resident line)"
  done
} >> "$LOG" 2>&1
