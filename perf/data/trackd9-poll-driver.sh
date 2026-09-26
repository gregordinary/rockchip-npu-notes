#!/bin/bash
# Is the pinning lever the LITTLE CORES, or is it ggml's POLL LOOP?
#
# The pinning arm measured taskset 0xf0 -t 4 at 1.126x over six passes, and its instruction
# accounting is the reason this arm exists: pinning removes 9.66e10 A55 instructions and 88 A55
# core-seconds, and the A76 count does not rise to absorb them. Total instructions fall 20% for
# the same tokens. That is the shape of work that was never useful.
#
# ggml's own threadpool is the candidate. ggml_graph_compute_poll_for_work spins
# 1024*128*poll rounds of `yield` before falling back to a mutex wait, and llama.cpp's default
# poll is 50 -- about 6.5e6 spins per worker per work item. On a graph whose heavy matmuls go to
# the NPU, the CPU workers have long nothing to do, and that is exactly when the poll loop runs.
#
# THE 2x2 IS THE POINT. --poll 0 attacks the spin without giving up any core; -t 4 on the A76s
# gives up half the cores to attack the placement. Running both axes crossed says which one
# carries the win and whether they compose. A recommendation of `--poll 0` is strictly better
# than one that halves a user's thread set, so the arms have to be able to tell them apart.
#
# THE SIGN OF --poll 0 IS NOT OBVIOUS. Sleeping on a mutex costs a wakeup per work item, which
# is a loss on a CPU-bound graph and a win only where the wait is long. That is why it is a
# registered prediction and not an assumption.
set -u
OUTD=${OUTD:-$PWD/trackd-poll}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/qwen35/Qwen3.5-0.8B-F16.gguf"
export PINWRAP_BIN=${PINWRAP_BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export BIN=${BIN:-$MODELS/llama-bench-pinwrap.sh}
export TESTS=${TESTS:-"-p 2048 -n 0 -r 3"}
export PASSES=${PASSES:-6}
export OUT="$OUTD/qwen35-08b-f16-poll.md"
export ERRD="$OUTD/err"
export ARMS='unpinned||
poll0||--poll 0
pin76t4|PIN_MASK=0xf0|-t 4
pinpoll|PIN_MASK=0xf0|-t 4 --poll 0'
LOG="$OUTD/driver.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "NOT EXECUTABLE: $BIN" >&2; exit 2; }
[ -x "$PINWRAP_BIN" ] || { echo "MISSING: $PINWRAP_BIN" >&2; exit 2; }
"$PINWRAP_BIN" --help 2>&1 | grep -q -- "--poll" || { echo "this llama-bench has no --poll" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- qwen35-08b-f16 poll-vs-pin 2x2  TESTS='$TESTS'  PASSES=$PASSES"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     arms: unpinned (base) | poll0 = --poll 0 | pin76t4 = PIN_MASK=0xf0 -t 4"
  echo "           pinpoll = both"
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 4 arms, RO on, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" qwen35-08b-f16-poll
  rc=$?
  echo "[$(date -Is)] rc=$rc ALL-DONE"
} >> "$LOG" 2>&1
