#!/bin/bash
# THE PINNED-PROTOCOL QUESTION. Does pinning BOTH arms shrink the ratio's per-pass spread?
# The ~6.5% per-pass paired-ratio sd on the 0.8B class is the binding constraint on the whole
# tuning matrix -- at three passes the ratio's se is ~3.8% and a knob under ~8% cannot resolve.
# If pinning tightens the RATIO, every future unit gets cheaper.
#
# READ THE CAUTION FIRST: the evidence that suggested this did NOT replicate. pin76t4's own ARM
# spread read 4.5% in the first pinning campaign and 11.9% in the second, same arm, same shape,
# six passes each. So there is no measured reason to expect this to work; it is here because the
# payoff is large and the arm is cheap.
#
# WHAT IS UNDER COMPARISON IS THE RATIO'S PER-PASS sd, NOT THE ARM'S SPREAD. The arm spread is a
# different statistic and it is the one that already failed to replicate. Pinning also moves the
# LEVEL, so the two campaigns are NOT comparable as levels -- only their sds are.
#
# THE UNIT is qwen35-08b-f16 stock against ROCKET_F16_RESIDENT=auto, which is the two-arm unit
# that currently straddles: 1.022x over seven passes at 1.109 0.894 1.059 1.093 1.007 1.046 0.944.
# A unit that already resolves could not show a tightening.
#
# CAMPAIGN=unpinned or CAMPAIGN=pinned selects which. Run BOTH, six passes each.
set -u
CAMPAIGN=${CAMPAIGN:?set CAMPAIGN=unpinned or CAMPAIGN=pinned}
OUTD=${OUTD:-$PWD/trackd-pinsd/$CAMPAIGN}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/qwen35/Qwen3.5-0.8B-F16.gguf"
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export PINWRAP_BIN=${PINWRAP_BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export BIN=${BIN:-$MODELS/llama-bench-pinwrap.sh}
export TESTS=${TESTS:-"-p 2048 -n 0 -r 3"}
export PASSES=${PASSES:-6}
export OUT="$OUTD/qwen35-08b-f16-$CAMPAIGN.md"
export ERRD="$OUTD/err"
case "$CAMPAIGN" in
  unpinned) export ARMS='stock||
f16res|ROCKET_F16_RESIDENT=auto|' ;;
  pinned)   export ARMS='stock|PIN_MASK=0xf0|-t 4
f16res|ROCKET_F16_RESIDENT=auto PIN_MASK=0xf0|-t 4' ;;
  *) echo "CAMPAIGN must be unpinned or pinned" >&2; exit 2 ;;
esac
LOG="$OUTD/driver.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "NOT EXECUTABLE: $BIN" >&2; exit 2; }
[ -x "$PINWRAP_BIN" ] || { echo "MISSING: $PINWRAP_BIN" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- qwen35-08b-f16 protocol campaign '$CAMPAIGN'  TESTS='$TESTS'  PASSES=$PASSES"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     arms: stock | f16res = ROCKET_F16_RESIDENT=auto"
  echo "     campaign 'pinned' applies PIN_MASK=0xf0 -t 4 to BOTH arms, so pinning is not a"
  echo "     difference WITHIN a campaign; what is compared across the two campaigns is the"
  echo "     per-pass paired-ratio sd, NOT the level and NOT the arm spread."
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: CAMPAIGN=$CAMPAIGN PASSES=$PASSES, 2 arms, RO on, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" "qwen35-08b-f16-$CAMPAIGN"
  rc=$?
  echo "[$(date -Is)] rc=$rc ALL-DONE"
} >> "$LOG" 2>&1
