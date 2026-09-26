#!/bin/bash
# Does the A76-pinning lever read across? The 0.8B f16 unit measured pin76 1.103x and
# pin76t4 1.126x over six rotated passes. That is one model, and one model's ratio does not
# read across to another on this board. This runs the other end of the matrix under the
# recipe the guide actually recommends: qwen35-9b with ROCKET_QUANT_RESIDENT=auto at the
# DEFAULT -ub, which is the unstacked form the seven-of-seven class rule ships.
#
# TWO THINGS DIFFER FROM THE 0.8B UNIT AT ONCE, deliberately and stated rather than hidden:
# the model is eleven times larger AND the config is the quant-resident recipe rather than
# f16-stock. This arm answers "does pinning survive where the lever would be used", which is
# the decision, not "is the effect a function of size alone", which would need a matched f16.
#
# READ THE ENGAGEMENT LINE PER ARM. Two arms of the same model are not necessarily at the
# same residency; the arms are comparable only if every one reports 200 resident / 0 streamed.
set -u
OUTD=${OUTD:-$PWD/trackd-pin9b}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/qwen35/Qwen3.5-9B-Q4_K_M.gguf"
export PINWRAP_BIN=${PINWRAP_BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export BIN=${BIN:-$MODELS/llama-bench-pinwrap.sh}
export TESTS=${TESTS:-"-p 2048 -n 0 -r 3"}
export PASSES=${PASSES:-3}
export OUT="$OUTD/qwen35-9b-qres-pin.md"
export ERRD="$OUTD/err"
export ARMS='unpinned|ROCKET_QUANT_RESIDENT=auto|
pin76|ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0|
pin76t4|ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0|-t 4'
LOG="$OUTD/driver.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "NOT EXECUTABLE: $BIN" >&2; exit 2; }
[ -x "$PINWRAP_BIN" ] || { echo "MISSING: $PINWRAP_BIN" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- qwen35-9b quant-resident, pinning intervention  TESTS='$TESTS'  PASSES=$PASSES"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     every arm carries ROCKET_QUANT_RESIDENT=auto at the default -ub (the unstacked recipe)"
  echo "     arms: unpinned | pin76 = PIN_MASK=0xf0 | pin76t4 = PIN_MASK=0xf0 -t 4"
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 3 arms, RO on, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" qwen35-9b-qres-pin
  rc=$?
  echo "[$(date -Is)] rc=$rc ALL-DONE"
} >> "$LOG" 2>&1
