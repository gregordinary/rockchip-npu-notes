#!/bin/bash
# Track D's INTERVENTION arm. The twelve rows before this one say the per-process spread on
# qwen35-08b-f16 is carried by l3ref_pki (rho -0.972) and by nothing upstream of the L3, and
# that the four fastest processes are exactly the four whose a55_inst_share is lowest. Those two
# columns co-vary and an observational design cannot order them; this one intervenes on the
# cluster split directly and reads what happens to the L3.
#
# WHY taskset AND NOT ROCKET_CPU_AFFINITY. librocketnpu already pins its pack and readback
# workers to the max-freq cluster (rocket_affinity.c: all four A76s here, cpuinfo_max_freq
# 2400000 against the A55s' 1800000), so the A55 instructions the readout sees are llama.cpp's
# own ggml threads. taskset is the knob that moves THOSE. That taskset cannot confine
# librocketnpu does not matter, because librocketnpu is already where the arm wants it.
#
# THE MASK TRAVELS AS A VARIABLE, NOT AS A COMMAND. bench-llm.sh expands its timed command as
# `env $envs ROCKET_LOG_STDERR=1 $BIN ...`, so `taskset 0xf0` written into an arm's env field
# eats that assignment and the arm dies rc=127 in 46 s with a header, a <!--PRED--> line and a
# full plausible <!--RO--> line already written. PIN_MASK plus llama-bench-pinwrap.sh is the
# route that works, and BIN points at the wrapper for every arm so it is not itself a difference.
#
# THREE ARMS, NOT TWO. llama-bench defaults to -t 8 on this board, so `taskset 0xf0` alone runs
# eight threads on four cores and changes threads-per-core as well as thread placement -- a
# faster pinned arm would then be compatible with "less contention" as well as with the L3
# story. pin76t4 holds threads-per-core at 1, as the unpinned arm has it, and costs ~9 minutes.
#
# READING IT: the load-bearing column is the ABSOLUTE a76_l3d_cache_refill, not l3ref_pki. Both
# arms run the same tokens, so the absolute count is comparable across them, while the density's
# denominator moves BY DESIGN here -- a76_inst_retired should rise ~26% in the pinned arms as the
# A55 work migrates, which alone would drop l3ref_pki by ~21% with no change in refills at all.
set -u
OUTD=${OUTD:-$PWD/trackd-pin}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/qwen35/Qwen3.5-0.8B-F16.gguf"
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export PINWRAP_BIN=${PINWRAP_BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export BIN=${BIN:-$MODELS/llama-bench-pinwrap.sh}
export TESTS=${TESTS:-"-p 2048 -n 0 -r 3"}
export PASSES=${PASSES:-6}
export OUT="$OUTD/qwen35-08b-f16-pin.md"
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
  echo "<!-- qwen35-08b-f16 pinning intervention  TESTS='$TESTS'  PASSES=$PASSES"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     arms: unpinned  (base; llama-bench default -t 8 over all 8 CPUs)"
  echo "           pin76     PIN_MASK=0xf0 -- the four A76s, -t 8 (2:1 oversubscribed)"
  echo "           pin76t4   PIN_MASK=0xf0 -t 4 -- threads-per-core held at 1"
  echo "     librocketnpu pins its own workers to the A76s in every arm."
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 3 arms, RO on, $(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" qwen35-08b-f16-pin
  rc=$?
  echo "[$(date -Is)] rc=$rc ALL-DONE"
} >> "$LOG" 2>&1
