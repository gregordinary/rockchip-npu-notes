#!/bin/bash
# f16-resident-readout.sh [MODEL_KEY ...]   -- what ROCKET_F16_RESIDENT=auto actually MOVES
#
# The matrix's f16 units are two arms each and three passes per unit, and unit 2 spent seven
# passes to land unresolved for a reason that needed no timing at all: on a 0.8B model the
# default already residents every K<=2048 weight, so the knob moved 126 resident weights to
# 150. The teardown `[f16-resident]` line says that directly. This runs ONE warm-up-sized
# process per arm and prints the split, so a unit whose knob has nothing to move is dropped
# for a measured reason in a couple of minutes rather than in an hour.
#
# TWO THINGS THIS NEEDS THAT bench-llm.sh's OWN WARM-UP DOES NOT HAVE:
#   ROCKET_LOG_STDERR=1 and stderr KEPT -- the warm-up sends both to /dev/null; and
#   a prefill long enough to be OFFLOADED at all. A ROCKET_MIN_M refusal (default 128) and a
#   knob with nothing to move both print zero, so CONFIRM THE STOCK ARM'S COUNT IS NON-ZERO
#   before reading a delta as small. -p 512 clears the default floor with room.
#
# Usage: MODELS=/path/to/models bash f16-resident-readout.sh llama32-3b-f16 ministral3-3b-f16
set -u
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
BIN=${BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
OUTD=${OUTD:-$MODELS/f16-resident-readout}
PROMPT=${PROMPT:--p 512 -n 0 -r 1}
mkdir -p "$OUTD"

path_for() {
  case "$1" in
    qwen35-08b-f16)    echo qwen35/Qwen3.5-0.8B-F16.gguf ;;
    llama32-3b-f16)    echo llama32-3b/Llama-3.2-3B-Instruct-F16.gguf ;;
    ministral3-3b-f16) echo ministral3-3b/Ministral-3-3B-Instruct-2512-F16.gguf ;;
    phi4mini-f16)      echo phi4mini/Phi-4-mini-instruct-F16.gguf ;;
    gemma4-12b-f16)    echo gemma4/gemma-4-12b-it-F16.gguf ;;
    *) echo "" ;;
  esac
}

read_arm() { # $1 key  $2 label  $3 extra env
  local m; m=$MODELS/$(path_for "$1")
  local err=$OUTD/$1.$2.err
  [ -f "$m" ] || { echo "MISSING $m"; return 1; }
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  # shellcheck disable=SC2086
  env GGML_BACKEND_PATH="$SO" ROCKET_LOG_STDERR=1 $3 "$BIN" -m "$m" $PROMPT >/dev/null 2>"$err"
  grep -h "f16-resident" "$err" | sed "s/^/    /"
}

for key in "$@"; do
  [ -n "$(path_for "$key")" ] || { echo "unknown key $key"; continue; }
  echo "== $key =="
  echo "  stock:"; read_arm "$key" stock ""
  echo "  ROCKET_F16_RESIDENT=auto:"; read_arm "$key" auto "ROCKET_F16_RESIDENT=auto"
done
echo
echo "stderr kept under $OUTD"
