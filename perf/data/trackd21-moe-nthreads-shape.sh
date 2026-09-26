#!/bin/bash
# THE SHAPE OF THE MoE ROUTE'S ROCKET_N_THREADS CURVE BETWEEN THE TWO MEASURED ENDS.
#
# WHAT IS ALREADY SETTLED, AND WHAT THIS DOES NOT RE-ASK. On this route the knob cannot buy
# placement: the pre-flight announces `Bound by RAM` at a byte-identical 63 admitted stacks /
# 24529 MB RAM / 15946 MB IOVA at nw=5 and nw=8, across a window doubled from 19200 to 30720 MB
# [HW sweep 2026-09-02, RK1; raw in ro-session/trackd17-moe-preflight.md]. What the knob COSTS
# was then measured on the same cell and it is a GAIN: nt8/nt5 = 1.0274x, 20.1 se above 1.00,
# three rotated passes [ro-session/trackd17c-moe-nthreads-cost.md]. Neither question is re-asked.
#
# WHAT IS OPEN. Only the SHAPE between 5 and 8, and it decides whether the default should move
# and by how much. Two shapes are live and they prescribe different defaults:
#
#   RAMP     -- more fds fan the expert GEMMs wider, which is continuous in the worker count.
#               nt6 and nt7 land between the ends and every step up pays.
#   STEP     -- `rocket_pin_worker_based` pins worker i to big core (base+i) mod 4
#               [source-confirmed, rocket_affinity.c], and the RK3588 has four A76s. 5, 6 and 7
#               all leave a core doubled; only 8 is uniform. Then nt6 and nt7 read ~1.000 and the
#               whole 2.7% belongs to 8 alone.
#
# The prediction was registered (2026-09-02) BEFORE this ran: ramp, with nt6/nt5
# 1.009 and nt7/nt5 1.018, against the step rival at 1.000 +- 0.005 for both.
#
# THE ARMS. Four, so the ratios are paired WITHIN a pass against the default. nt5 is spelled as
# an ABSENCE so a default move is visible. nt8 is re-run rather than read across from the
# recorded cell: a ratio taken in a different pass is not paired, and re-running it also
# reproduces the endpoint under this session's board state.
#
# NO CAP APPLIES: this prices a default that ships rather than proposing a lever.
#
# PASS BUDGET. Three. The budget is a property of the CONTRAST and this one has a measured sd:
# the nt8/nt5 paired ratio read 1.0289 / 1.0287 / 1.0247, sd 0.0024, se 0.0014 over three passes.
# The smaller of the two predicted steps is 0.9 pp, about 6 se, so three passes resolve either
# shape. Extend only if a pass comes back outside that spread.
#
# READ THE `[moe-int8]` STACK COUNT PER ARM. It must be 63 in every arm or the arms are not
# comparable and the ladder is measuring placement, not fan-out.
set -u
OUTD=${OUTD:-$PWD/trackd21-moe-ntshape}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/gpt-oss-20b/gpt-oss-20b-mxfp4.gguf"
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export BIN=${BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export TESTS=${TESTS:-"-p 2048 -n 0 -r 3"}
export PASSES=${PASSES:-3}
export OUT="$OUTD/gptoss-20b-nthreads-shape.md"
export ERRD="$OUTD/err"
export ARMS='nt5||-b 2048 -ub 2048
nt6|ROCKET_N_THREADS=6|-b 2048 -ub 2048
nt7|ROCKET_N_THREADS=7|-b 2048 -ub 2048
nt8|ROCKET_N_THREADS=8|-b 2048 -ub 2048'
LOG="$OUTD/trackd21.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "MISSING BIN: $BIN" >&2; exit 2; }
[ -f "$SO" ]    || { echo "MISSING SO: $SO" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- gpt-oss-20b MXFP4, the ROCKET_N_THREADS curve between the two measured ends"
  echo "     TESTS='$TESTS'  PASSES=$PASSES  args='-b 2048 -ub 2048' on every arm"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     every arm is RAM-bound: the pre-flight announces Bound by RAM at 63 stacks at 5 and 8"
  echo "     ratios are paired WITHIN a pass against nt5; arm order rotated by one each pass"
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 4 arms, RO on, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" gptoss-20b-nthreads-shape
  rc=$?
  echo "[$(date -Is)] rc=$rc TRACKD21-ALL-DONE"
} >> "$LOG" 2>&1
