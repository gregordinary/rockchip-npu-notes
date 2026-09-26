#!/bin/bash
# READING ROCKET_FA_PROFILE ON HARDWARE: WHERE THE FLASH-ATTENTION COMPUTE SEGMENT GOES.
#
# WHAT IS OPEN. The handler's bracket splits the FA offload into gather / compute / scatter, and
# threading the two host walks is settled and closed. The compute segment -- 17.4% of the pinned
# prefill wall -- mixes the device's QK/AV with the HOST mask and softmax, and nothing has split
# them. `expf` is 7.72% of A76 cycles = 12 core-s per prefill; its borrowed-rate cap of 1.97% is a
# FLOOR and the 17.4% is the ceiling, so a vectorised exp is priced in a bracket nearly ten times
# wide. ROCKET_FA_PROFILE closes it.
#
# WHY BOTH KNOBS IN ONE PROCESS. ROCKET_FA_TIMING gives the handler's bracket (the compute segment
# in WALL seconds) and ROCKET_FA_PROFILE gives the driver's split of that segment. Read from the
# SAME process they need no imported constant: the ratio and the wall it multiplies were measured
# on one build, on one graph, in one run. Importing the published 17.4% instead would carry a
# number across twelve commits of `.so`, and the one absolute that has already moved that way --
# `G_1`, 5.44 -> 5.80 s -- moved 6.6%.
#
# HOW THE PROFILE IS READ, AND HOW IT IS NOT. Two lines are printed. The `total` line's buckets are
# worker-thread intervals summed over CONCURRENT workers; they exceed the wall by about the worker
# count and are a share of nothing. The `max-range` line is the largest single range's OWN buckets
# -- one thread, one interval, and the range the dispatch thread waited for -- and it is the line
# to divide. The quantity is `softmax` / `max-range` applied to this run's own compute segment.
# A bucket also counts descheduled time, so every term is an UPPER bound: enough to close a lever,
# not enough to size one.
#
# THE PREDICTION [registered 2026-09-07, before any device read]. `softmax`/`max-range`
# = 0.18-0.40, a cap on deleting `expf` of 3.1-7.0% of the pinned wall, and the ordering inside
# `max-range` is qk+av > softmax > mask > gather > scatter. RIVAL: below 0.18 the range is device
# wait and the lever is closed at its 1.97% floor. SECOND RIVAL: above 0.40 the host softmax is
# the range's largest term and a vectorised exp is the largest remaining host lever.
#
# THE CONTROL IS THE SECOND ARM. An instrument that perturbs its own subject is the failure mode,
# and the driver's own flash_attn gate has already been run both ways on both FA paths. This arm
# pair is the same question asked of the REAL workload: if prof_on and prof_off differ in wall
# beyond the pass spread, the profile is not free and its own numbers are suspect.
#
# CHECK THE <!--DATA--> ROW COUNT: 2 arms x 3 passes = 6 rows. Then check that every arm's .err
# carries a `ROCKET FA total` line with the SAME op count, and that every arm streamed.
set -u
OUTD=${OUTD:-$PWD/trackd36-faprofile12b}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/gemma4/gemma-4-12b-it-F16.gguf"
export PINWRAP_BIN=${PINWRAP_BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export BIN=${BIN:-$MODELS/llama-bench-pinwrap.sh}
export TESTS=${TESTS:-"-p 2048 -n 0 -r 2"}
export PASSES=${PASSES:-3}
export OUT="$OUTD/gemma4-12b-f16-faprofile.md"
export ERRD="$OUTD/err"
export ARMS='prof_off|ROCKET_FA_TIMING=1 PIN_MASK=0xf0|-t 4 -v
prof_on|ROCKET_FA_TIMING=1 ROCKET_FA_PROFILE=1 PIN_MASK=0xf0|-t 4 -v'
LOG="$OUTD/trackd36.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "NOT EXECUTABLE: $BIN" >&2; exit 2; }
[ -x "$PINWRAP_BIN" ] || { echo "MISSING: $PINWRAP_BIN" >&2; exit 2; }
[ -f "$SO" ]    || { echo "MISSING: $SO" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- gemma4-12b F16, PINNED (PIN_MASK=0xf0 -t 4), streamed: reading ROCKET_FA_PROFILE"
  echo "     TESTS='$TESTS'  PASSES=$PASSES  every arm -v and ROCKET_FA_TIMING=1, so neither is a difference"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     so=$(md5sum "$SO" | cut -d' ' -f1)"
  echo "     librocketnpu=$(md5sum /usr/local/lib/librocketnpu.a 2>/dev/null | cut -d' ' -f1)"
  echo "     The deliverable is the profile's max-range line from the prof_on arms; the wall"
  echo "     comparison is the CONTROL on whether the instrument perturbs its own subject."
  echo "     PREDICTION 2026-09-07b: softmax/max-range = 0.18-0.40, cap 3.1-7.0% of the pinned wall,"
  echo "       ordering qk+av > softmax > mask > gather > scatter. Below 0.18 the device dominates"
  echo "       and the lever is closed at 1.97%; above 0.40 the host softmax is the largest term."
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 2 arms, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" gemma4-12b-f16-faprofile
  rc=$?
  echo "[$(date -Is)] rc=$rc TRACKD36-ALL-DONE"
  echo "--- FA timing lines, one per arm per pass ---"
  for f in "$ERRD"/*.err; do
    printf '%s: ' "$(basename "$f" .err)"
    grep -a "ROCKET FA total" "$f" | tail -1 || echo "(no FA total line)"
  done
  echo "--- FA profile lines, prof_on arms only ---"
  for f in "$ERRD"/*.err; do
    grep -aq "FA profile" "$f" || continue
    printf '%s:\n' "$(basename "$f" .err)"
    grep -a "FA profile" "$f"
  done
  echo "--- residency control: every arm must report 0 resident ---"
  for f in "$ERRD"/*.err; do
    printf '%s: ' "$(basename "$f" .err)"
    grep -aoE '\[f16-resident\][^\n]*' "$f" | tail -1 || echo "(no resident line)"
  done
} >> "$LOG" 2>&1
echo "TRACKD36-DONE" >> "$LOG"
