#!/bin/bash
# THE 2x2 IN PINNING AND RESIDENCY, ON ONE MODEL. Two questions, one campaign.
#
# QUESTION 1, the de-confounder the plan asked for and could not schedule. The pinning lever
# reads 1.103x/1.126x on a resident 0.8B, 1.059x/1.062x on a resident 9B and 1.046x/1.062x on a
# STREAMED 12B. Those cells differ in residency AND in size at once, so neither separates them.
# A resident and a streamed configuration of the SAME GGUF does. That arm was believed
# unavailable because "Gemma-4-12B F16 cannot be held resident on this board" -- which is false:
# ROCKET_F16_RESIDENT=auto places 286 of 328 weights (18078 MB), 87% resident, and admission
# stops at the NPU IOVA window rather than at RAM [HW readout 2026-09-01, f16-resident-readout.sh].
#
# QUESTION 2, a matrix gap. The f16 class has no 12B row. The three K=3072 3B-class units go from
# no route at all to fully resident and gain 1.078-1.084x, a remarkably tight band; this model
# goes to 87% rather than 100%, so its remainder still pays the per-call pack.
#
# STOCK IS THE STREAMED ARM HERE, AND THAT IS MEASURED, NOT ASSUMED. The stock arm residents ZERO
# weights: this model's K exceeds the default prepack gate, so nothing is offered to the route.
# An absent [f16-resident] line is not a zero on its own, which is why the readout was run first.
#
# READ THE PLACED FRACTION PER ARM. Placement varies run to run on a partly-placed model, so an
# arm that placed 87% and one that placed 79% are not the same configuration. Every resident arm
# prints its own count; a magnitude read across arms that placed differently is not a ratio.
set -u
OUTD=${OUTD:-$PWD/trackd-res12b}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/gemma4/gemma-4-12b-it-F16.gguf"
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export PINWRAP_BIN=${PINWRAP_BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export BIN=${BIN:-$MODELS/llama-bench-pinwrap.sh}
export TESTS=${TESTS:-"-p 2048 -n 0 -r 3"}
export PASSES=${PASSES:-3}
export OUT="$OUTD/gemma4-12b-f16-res2x2.md"
export ERRD="$OUTD/err"
export ARMS='stream_unpin||
stream_pin|PIN_MASK=0xf0|-t 4
res_unpin|ROCKET_F16_RESIDENT=auto|
res_pin|ROCKET_F16_RESIDENT=auto PIN_MASK=0xf0|-t 4'
LOG="$OUTD/driver.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "NOT EXECUTABLE: $BIN" >&2; exit 2; }
[ -x "$PINWRAP_BIN" ] || { echo "MISSING: $PINWRAP_BIN" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- gemma4-12b F16 2x2 in pinning and residency  TESTS='$TESTS'  PASSES=$PASSES"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     arms: stream_unpin | stream_pin = PIN_MASK=0xf0 -t 4"
  echo "           res_unpin    = ROCKET_F16_RESIDENT=auto"
  echo "           res_pin      = ROCKET_F16_RESIDENT=auto + PIN_MASK=0xf0 -t 4"
  echo "     stock residents ZERO weights on this model (K above the default prepack gate),"
  echo "     so the two stream_* arms are the streamed configuration, measured not assumed."
  echo "     The knob reaches 286 of 328 weights (18078 MB, 87%); the rest still stream."
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 4 arms, RO on, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" gemma4-12b-f16-res2x2
  rc=$?
  echo "[$(date -Is)] rc=$rc ALL-DONE"
} >> "$LOG" 2>&1
