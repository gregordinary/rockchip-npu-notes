#!/bin/bash
# THE FIRST NON-NESTED THIRD POINT ON THE 9B: UNSTACKED QUANT RESIDENCY, WITH ITS OWN `a`.
#
# WHAT IS SETTLED. `qwen35-9b` Q4_K_M reads `a/phi` = 0.602 on quant residency (phi 0.6645 off the
# rep-count regression, a = 0.4003) and 0.591 on `-b 2048 -ub 2048` (phi 0.4958, a = 0.2928). The
# two are NESTED -- the residency arm IS the micro-batch arm with residency on top -- so their 1.9%
# agreement is a two-point collinearity test through the origin with no residual degree of freedom
# [tuning-matrix.md; ro-session/trackd29-busy-intercept-9b.md; trackd16, trackd20].
#
# THE ARM. `ROCKET_QUANT_RESIDENT=auto` at the DEFAULT `-ub` -- the recommended recipe,
# published at 1.752x (1.737-1.771) [ro-session/qwen35-9b-qres512.md], and not nested with the
# `-ub` knob. Six PINNED arms give the rep-count regression (`busy_tot = intercept + (r+1)*slope`,
# phi = 1 - slope_q/slope_s uncontaminated, ingest = intercept difference), and two UNPINNED
# arms at -r 2 give this campaign's own `a` -- imported `a` was the previous design's weak point,
# and the published ratio is a reproduction band that a different interleave can move 2.5 se.
# The pinned r2 pair plus the unpinned r2 pair is also a complete 2x2 at -r 2: a seventh
# interaction cell for free.
#
# THE PREDICTION, DERIVED FROM TWO EXISTING CELLS AND ONE MECHANISM [registered 2026-09-03].
# Stock at `-ub 512` runs FOUR micro-batches per 2048-token prefill and re-dequantizes and re-packs
# every weight per micro-batch. Write D for those four dequant+pack passes in core-seconds and R
# for whatever else `-ub 2048` changes in host work. Then, as fractions of the stock arm's per-rep
# host CPU C_s:
#     `-b 2048 -ub 2048`        removes 0.75 D + R  = 0.4958 C_s   (trackd20, pinned pair)
#     `-ub 2048` + residency    removes      D + R  = 0.6645 C_s   (trackd29 slope)
# so D = 0.675 C_s and R = -0.010 C_s (the -ub knob adds ~1% of host work). Unstacked residency
# removes D alone: phi_u = 0.675, band 0.667-0.682 (the derived value's own 1 se, from se 0.0015
# and 0.0011 on its inputs), and the measurement's own se should be ~0.0015 as before.
# Ingest: the SAME 200 weights / 13184 MB at 4.53 ms per resident MB = 59.7 core-s (56-63, the
# trackd29 per-pass spread). K_u unpinned = 1.752 (1.72-1.79, a REPRODUCTION band). Then
# a_u = 0.429, a/phi = 0.636 (0.62-0.65), q = 4.18 (4.0-4.35), against the nested pair's 4.46 --
# so the third point sits ABOVE the nested line a = 0.602 phi by +0.023 in a, four to six se, and
# what that residual measures is the wall term of the `-ub` component the nested pair shares.
# RIVAL A: phi_u = 0.6645, the plan's "the removed work is the same dequant and pack" -- 7 se
# from the point at the measurement's se, 1.4 se at the derivation's. RIVAL B: phi_u > 0.69 says
# R is larger and negative than the two cells imply, i.e. `-ub 2048` adds real host work at the
# resident micro-batch. RIVAL C: K_u outside 1.72-1.79 says board state moved and the published
# row is not this board's; the campaign's own a is used either way.
# The 2x2 at -r 2 predicts, under the published model I = (1-a)(1-b)/(1-a-b+b*phi) with the
# campaign's own b: about 0.959 at b = 0.09. Recorded with its se; three passes on a 9B
# interaction (sd 1.38%) resolve it from the null, not from the model.
#
# NOT A PERF LEVER, SO NO CAP APPLIES. The output is a denominator and a residual.
#
# CONTROLS. Every qres512 arm must report 200 / 13184 MB, every stock arm none. Placement must be
# constant across rep counts or the regression fits three configurations. Eight arms rotate over
# three passes, so each arm sees three of eight positions; the position effect is +-0.3% of t/s
# [trackd23], ~0.2% relative on phi, and the interaction carries it unbalanced -- read the
# interaction's se, not its point.
#
# WHAT A GREEN RESULT WOULD NOT SHOW. Three points on one unit; nothing about another unit; `q`
# has never been measured on a model outside these two.
#
# CHECK THE <!--DATA--> ROW COUNT AFTER PASS 1: 8 arms, so 8 rows.
set -u
OUTD=${OUTD:-$PWD/trackd31-qres512-9b}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/qwen35/Qwen3.5-9B-Q4_K_M.gguf"
export PINWRAP_BIN=${PINWRAP_BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export BIN=${BIN:-$MODELS/llama-bench-pinwrap.sh}
# -r lives in each arm's args, so TESTS must NOT carry one.
export TESTS=${TESTS:-"-p 2048 -n 0"}
export PASSES=${PASSES:-3}
export OUT="$OUTD/qwen35-9b-qres512-third-point.md"
export ERRD="$OUTD/err"
export ARMS='stock_r1|PIN_MASK=0xf0|-t 4 -r 1
stock_r2|PIN_MASK=0xf0|-t 4 -r 2
stock_r4|PIN_MASK=0xf0|-t 4 -r 4
qres512_r1|ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0|-t 4 -r 1
qres512_r2|ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0|-t 4 -r 2
qres512_r4|ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0|-t 4 -r 4
stock_u_r2||-r 2
qres512_u_r2|ROCKET_QUANT_RESIDENT=auto|-r 2'
LOG="$OUTD/trackd31.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "NOT EXECUTABLE: $BIN" >&2; exit 2; }
[ -x "$PINWRAP_BIN" ] || { echo "MISSING: $PINWRAP_BIN" >&2; exit 2; }
[ -f "$SO" ]    || { echo "MISSING: $SO" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- qwen35-9b Q4_K_M, UNSTACKED quant residency (ROCKET_QUANT_RESIDENT=auto at the default -ub)"
  echo "     TESTS='$TESTS'  PASSES=$PASSES  (-r lives in each arm's args)"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     six PINNED arms (PIN_MASK=0xf0 -t 4) at -r 1/2/4: busy_tot = intercept + (r+1)*slope"
  echo "       phi_u = 1 - slope_qres512/slope_stock ; ingest = intercept_qres512 - intercept_stock"
  echo "     two UNPINNED arms at -r 2: K_u = qres512_u_r2/stock_u_r2 paired within a pass; a_u = 1 - 1/K_u"
  echo "     2x2 at -r 2: interaction = (qres512_r2/stock_r2) / (qres512_u_r2/stock_u_r2)"
  echo "     PREDICTION 2026-09-03e: phi_u = 0.675 (0.667-0.682), ingest 59.7 core-s (56-63),"
  echo "       K_u 1.752 (1.72-1.79, reproduction band), a/phi 0.636 (0.62-0.65), q 4.18 (4.0-4.35);"
  echo "       the third point sits +0.023 in a ABOVE the nested line a = 0.602 phi."
  echo "       RIVAL A phi_u = 0.6645 (same removed work as stacked). RIVAL B phi_u > 0.69."
  echo "       RIVAL C K_u outside 1.72-1.79 (board state moved)."
  echo "     CONTROL: every qres512 arm 200 weights / 13184 MB; every stock arm zero"
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 8 arms, RO on, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" qwen35-9b-qres512-third-point
  rc=$?
  echo "[$(date -Is)] rc=$rc TRACKD31-ALL-DONE"
  echo "--- placement control: every qres512 arm 200/13184MB, every stock arm zero ---"
  for f in "$ERRD"/*.err; do
    printf '%s: ' "$(basename "$f")"
    grep -oE '\[(f16|quant)-resident\][^\n]*' "$f" | tail -1 || echo "(no resident line)"
  done
} >> "$LOG" 2>&1
