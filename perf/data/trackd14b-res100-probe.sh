#!/bin/bash
# THE 100%-RESIDENT ARM AT -p 2048: AN OOM BUDGET AND A PROFILE, BEFORE ANY PASSES ARE SPENT.
#
# WHY A PROBE AND NOT THE CAMPAIGN. `ROCKET_N_THREADS=8` + `ROCKET_QUANT_RESIDENT_RESERVE_MB=6144`
# places 328 of 328 weights (20790MB, 0 streamed), reproduced twice -- but only at `-p 512 -n 0
# -r 1`, where it left MemAvailable at 6542MB against its own 6144MB floor on a board with NO
# SWAP. A -p 2048 prefill carries a KV cache four times that size into the same headroom, and an
# OOM kill voids the arm and costs the session. So the headroom is read ONCE, unpaired, before
# three passes are committed to it.
#
# WHAT ELSE IT BUYS. The profiler in this configuration says packB should be ZERO -- if it is
# not, the arm is not the configuration it is labelled, and no wall ratio taken against it means
# what it says. It also gives the first (unreplicated) wall reading, which sizes the campaign.
#
# WHY THE CAP CANNOT BE READ FROM packB ALONE [source-confirmed, rocket_matmul.c]. mm_compute*
# adds its `t_pack` argument to g_prof.pack but to NEITHER packA nor packB, so the aggregate's
# `pack - packA - packB` is an UNCLASSIFIED input-scatter bucket carrying both routes' per-worker
# A work. Residency moves a call from the mt route (a full per-worker A scatter) to the prepacked
# route (one shared upstream scatter plus a per-worker memcpy), so it reaches that bucket too and
# the aggregate cannot split it by route. Differencing the two configurations' profiles can.
set -u
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/gemma4/gemma-4-12b-it-F16.gguf"
SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
BIN=${BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
OUTD=${OUTD:-$MODELS/trackd14b-res100}
TESTS=${TESTS:--p 2048 -n 0 -r 2}
ARMS=${ARMS:-'res100|ROCKET_N_THREADS=8 ROCKET_QUANT_RESIDENT_RESERVE_MB=6144'}
LOG="$OUTD/readout.md"

mkdir -p "$OUTD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "MISSING: $BIN" >&2; exit 2; }
[ -f "$SO" ]    || { echo "MISSING: $SO" >&2; exit 2; }

oom_count() { sudo dmesg 2>/dev/null | grep -icE "out of memory|oom-kill|oom_reaper"; }
OOM0=$(oom_count)

{
  echo "<!-- gemma4-12b F16 100%-resident probe at -p 2048   TESTS='$TESTS'"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     so=$(md5sum "$SO" | cut -d' ' -f1)  bin=$(md5sum "$BIN" | cut -d' ' -f1)"
  echo "     MemTotal=$(awk '/MemTotal/{print $2}' /proc/meminfo) kB  SwapTotal=$(awk '/SwapTotal/{print $2}' /proc/meminfo) kB"
  echo "     clk=$(sudo cat /sys/kernel/debug/clk/clk_summary | awk '/scmi_clk_npu/{print $5}') Hz"
  echo "     dmesg OOM lines before start: $OOM0"
  echo "-->"
  echo
} > "$LOG"

echo "[$(date -u +%FT%TZ)] res100 probe start" | tee -a "$LOG"

while IFS='|' read -r label envs; do
  [ -n "$label" ] || continue
  err="$OUTD/$label.err"; out="$OUTD/$label.md"; mem="$OUTD/$label.mem"
  echo | tee -a "$LOG"
  echo "### $label  env='$envs'  $(date -u +%T)" | tee -a "$LOG"

  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null 2>&1
  echo "    MemAvailable before: $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB" | tee -a "$LOG"

  # 2 s sampling: this arm's whole point is the low-water mark, and it approaches it fast
  ( while :; do awk '/MemFree|MemAvailable/{printf "%s ", int($2/1024)}' /proc/meminfo; echo; sleep 2; done ) > "$mem" &
  SAMP=$!

  # shellcheck disable=SC2086
  env GGML_BACKEND_PATH="$SO" ROCKET_LOG_STDERR=1 ROCKET_MM_PROFILE=1 ROCKET_F16_RESIDENT=auto $envs \
      "$BIN" -m "$MODEL" $TESTS -o md > "$out" 2>"$err"
  rc=$?
  kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null

  echo "    rc=$rc" | tee -a "$LOG"
  echo "    MemFree low-water: $(awk '{print $1}' "$mem" | sort -n | head -1) MB   MemAvailable low-water: $(awk '{print $2}' "$mem" | sort -n | head -1) MB   samples=$(wc -l < "$mem")" | tee -a "$LOG"
  grep -h "resident budget" "$err" | sed 's/^ *//;s/^/    /' | tee -a "$LOG"
  grep -h "f16-resident"    "$err" | sed 's/^ *//;s/^/    /' | tee -a "$LOG"
  grep -h "ROCKET profile"  "$err" | sed 's/^ *//;s/^/    /' | tee -a "$LOG"
  grep -hiE "No space left|CREATE_BO" "$err" | head -3 | sed 's/^ *//;s/^/    ALLOC: /' | tee -a "$LOG"
  echo "    --- llama-bench row ---" | tee -a "$LOG"
  grep -E "^\|" "$out" | sed 's/^/    /' | tee -a "$LOG"

  oom_now=$(oom_count)
  if [ "$oom_now" -gt "$OOM0" ]; then
    echo "    *** OOM KILL DETECTED -- VOID, and the campaign must NOT be launched at this shape ***" | tee -a "$LOG"
    sudo dmesg | grep -iE "out of memory|oom-kill" | tail -3 | sed 's/^/    /' | tee -a "$LOG"
    break
  fi
done <<< "$ARMS"

echo | tee -a "$LOG"
echo "[$(date -u +%FT%TZ)] ALL-DONE" | tee -a "$LOG"
