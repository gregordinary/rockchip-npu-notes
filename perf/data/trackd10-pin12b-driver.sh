#!/bin/bash
# THE 12B F16 DISAGREEMENT. A standing negative records whole-process `taskset 0xf0` as no win with
# prefill FLAT, measured on Gemma-4-12B F16. Prefill then measured 1.103x (pin76) and 1.126x
# (pin76t4) on qwen35-08b-f16 and 1.059x / 1.062x on qwen35-9b quant-resident, under rotated
# interleaved passes with a memory reset per arm. This runs the contrary cell under that protocol.
#
# WHY THE TWO MIGHT BOTH BE RIGHT. The pinning effect is entirely host-side -- the NPU half is
# identical in every arm of both campaigns. Both units that measured a win are 100% resident, so
# their host half is pack and readback. This one is 22.18 GiB and streams its weights from mmap
# per micro-batch, and that term is bandwidth-bound, where four cores issue fewer outstanding
# misses than eight. [expected] -- a mechanism story, not a measurement.
#
# WHY IT MIGHT BE METHOD. The recorded 12B cell has no raw evidence file and no recorded protocol
# anywhere in the workspace, and this board does not settle a sign at one process per arm.
#
# THREE ARMS, NOT TWO. The recorded negative is PLAIN `taskset 0xf0` with llama-bench's default
# -t 8, which is pin76 -- so pin76 is the faithful reproduction and pin76t4 is the arm that
# measured largest elsewhere. Dropping either leaves the disagreement unresolved in one direction.
# BIN points at the wrapper for every arm so the wrapper is not itself a difference between arms;
# the mask travels as PIN_MASK because a command wrapper in an arm's env field eats the
# assignment after it and dies rc=127 in 46 s looking like a fast arm.
#
# READING IT: read each arm's engagement line -- every arm must report the STREAMED configuration,
# because a resident arm is not the recorded cell. Then read pfn_zero_frac (want 0.0000) and
# pmu_enabled (want 100) before quoting any RO column, and the absolute a76_l3d_cache_refill
# rather than l3ref_pki, whose denominator moves by design here.
set -u
OUTD=${OUTD:-$PWD/trackd-pin12b}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/gemma4/gemma-4-12b-it-F16.gguf"
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export PINWRAP_BIN=${PINWRAP_BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export BIN=${BIN:-$MODELS/llama-bench-pinwrap.sh}
export TESTS=${TESTS:-"-p 2048 -n 0 -r 3"}
export PASSES=${PASSES:-3}
export OUT="$OUTD/gemma4-12b-f16-pin.md"
export ERRD="$OUTD/err"
export ARMS='unpinned||
pin76|PIN_MASK=0xf0|
pin76t4|PIN_MASK=0xf0|-t 4'
LOG="$OUTD/driver.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "NOT EXECUTABLE: $BIN" >&2; exit 2; }
[ -x "$PINWRAP_BIN" ] || { echo "MISSING: $PINWRAP_BIN" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- gemma4-12b F16 pinning intervention  TESTS='$TESTS'  PASSES=$PASSES"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     THE CONTRARY CELL: a standing negative recorded taskset 0xf0 as prefill-FLAT on this model,"
  echo "     with no raw evidence file and no recorded protocol. This is that cell under the"
  echo "     rotated-interleaved protocol that found the 1.06-1.13x lever on two other models."
  echo "     arms: unpinned  (base; llama-bench default -t 8 over all 8 CPUs)"
  echo "           pin76     PIN_MASK=0xf0 -- the four A76s, -t 8; the FAITHFUL reproduction"
  echo "           pin76t4   PIN_MASK=0xf0 -t 4 -- threads-per-core held at 1"
  echo "     22.18 GiB streams from mmap in every arm; librocketnpu pins its own workers to"
  echo "     the A76s in every arm."
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 3 arms, RO on, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" gemma4-12b-f16-pin
  rc=$?
  echo "[$(date -Is)] rc=$rc ALL-DONE"
} >> "$LOG" 2>&1
