#!/bin/bash
# WHAT DOES THE PER-CALL WEIGHT PACK COST ON THE 12B -- the CAP read that gates the
# "what does full residency buy" campaign, and a readout rather than a timed arm.
#
# THE TERM. At 87% resident, `ROCKET_F16_RESIDENT=auto` places 286 of 328 weights and the
# other 42 re-pack into NPU tiles on EVERY call. Full residency (ROCKET_N_THREADS=8 +
# ROCKET_QUANT_RESIDENT_RESERVE_MB=6144, which places 328/328) removes exactly that, so what
# it can buy is capped at 100% of that term's share of the wall.
#
# WHY packB IS THE RIGHT BUCKET AND THE WHOLE OF IT. `mm_prof_add_pack_b` is called from ONE
# site, rocket_prepacked.c's rkw_thread, and only on the streaming branch (`t->segs || t->B`);
# the resident build's own pack (rocket_prepacked.c:237) calls the same mm_pack_weights* but
# does NOT feed the profiler. So in a resident arm packB is the streamed remainder's per-call
# pack and nothing else. The weight's cache-sync (PREP_BO/FINI_BO) and the source-page faults
# both happen inside mm_pack_weights, so they are inside this timer too.
#
# THE DENOMINATOR. llama-bench runs ONE full-n_prompt warmup prefill before -r reps
# [source-confirmed, tools/llama-bench/llama-bench.cpp:2352-2358], so a -r N run does N+1
# prefills and the profiler accumulates over all of them. Per-prefill packB = packB/(N+1);
# per-prefill wall = n_prompt / (reported t/s). The resident build lands in the warmup, and it
# is not in packB, so the buckets are uniform across the N+1.
#
# ALSO READ: whether the profile's accounting CLOSES. A bucket that ends before the teardown
# attributes real time to nothing, and an unattributed remainder is where a cost hides.
#
# MEMORY. Arm A runs at the DEFAULT reserve and carries no OOM risk. It also reads the KV
# cache size and the low-water MemAvailable at -p 2048, which is what the 100%-resident arm's
# OOM budget needs and what the -p 512 readouts could not supply.
set -u
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/gemma4/gemma-4-12b-it-F16.gguf"
SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
BIN=${BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
OUTD=${OUTD:-$MODELS/trackd14-packcap}
TESTS=${TESTS:--p 2048 -n 0 -r 2}
ARMS=${ARMS:-'res87|'}
LOG="$OUTD/readout.md"

mkdir -p "$OUTD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "MISSING: $BIN" >&2; exit 2; }
[ -f "$SO" ]    || { echo "MISSING: $SO" >&2; exit 2; }

oom_count() { sudo dmesg 2>/dev/null | grep -icE "out of memory|oom-kill|oom_reaper"; }
OOM0=$(oom_count)

{
  echo "<!-- gemma4-12b F16 per-call weight-pack CAP readout   TESTS='$TESTS'"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     so=$(md5sum "$SO" | cut -d' ' -f1)  bin=$(md5sum "$BIN" | cut -d' ' -f1)"
  echo "     MemTotal=$(awk '/MemTotal/{print $2}' /proc/meminfo) kB  SwapTotal=$(awk '/SwapTotal/{print $2}' /proc/meminfo) kB"
  echo "     clk=$(sudo cat /sys/kernel/debug/clk/clk_summary | awk '/scmi_clk_npu/{print $5}') Hz"
  echo "     gov=$(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor)"
  echo "     dmesg OOM lines before start: $OOM0"
  echo "-->"
  echo
} > "$LOG"

echo "[$(date -u +%FT%TZ)] packcap readout start" | tee -a "$LOG"

while IFS='|' read -r label envs; do
  [ -n "$label" ] || continue
  err="$OUTD/$label.err"; out="$OUTD/$label.md"; mem="$OUTD/$label.mem"
  echo | tee -a "$LOG"
  echo "### $label  env='$envs'  $(date -u +%T)" | tee -a "$LOG"

  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null 2>&1
  echo "    MemAvailable before: $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB" | tee -a "$LOG"

  # low-water sampler: a local /proc read every 5 s, no ssh, negligible against a ~100 s prefill
  ( while :; do awk '/MemFree|MemAvailable/{printf "%s ", int($2/1024)}' /proc/meminfo; echo; sleep 5; done ) > "$mem" &
  SAMP=$!

  # shellcheck disable=SC2086
  env GGML_BACKEND_PATH="$SO" ROCKET_LOG_STDERR=1 ROCKET_MM_PROFILE=1 ROCKET_F16_RESIDENT=auto $envs \
      "$BIN" -m "$MODEL" $TESTS -o md > "$out" 2>"$err"
  rc=$?
  kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null

  echo "    rc=$rc" | tee -a "$LOG"
  echo "    MemFree low-water: $(awk '{print $1}' "$mem" | sort -n | head -1) MB   MemAvailable low-water: $(awk '{print $2}' "$mem" | sort -n | head -1) MB   samples=$(wc -l < "$mem")" | tee -a "$LOG"
  grep -hE "KV self size|kv_unified|n_ctx +=|size = .*MiB" "$err" | head -6 | sed 's/^ *//;s/^/    LLAMA: /' | tee -a "$LOG"
  grep -h "resident budget"   "$err" | sed 's/^ *//;s/^/    /' | tee -a "$LOG"
  grep -h "f16-resident"      "$err" | sed 's/^ *//;s/^/    /' | tee -a "$LOG"
  grep -h "ROCKET profile"    "$err" | sed 's/^ *//;s/^/    /' | tee -a "$LOG"
  grep -h "ROCKET int8 profile" "$err" | sed 's/^ *//;s/^/    /' | tee -a "$LOG"
  grep -hiE "No space left|CREATE_BO" "$err" | head -3 | sed 's/^ *//;s/^/    ALLOC: /' | tee -a "$LOG"
  echo "    --- llama-bench row ---" | tee -a "$LOG"
  grep -E "^\|" "$out" | sed 's/^/    /' | tee -a "$LOG"

  oom_now=$(oom_count)
  if [ "$oom_now" -gt "$OOM0" ]; then
    echo "    *** OOM KILL DETECTED during this arm -- VOID, stopping ***" | tee -a "$LOG"
    sudo dmesg | grep -iE "out of memory|oom-kill" | tail -3 | sed 's/^/    /' | tee -a "$LOG"
    break
  fi
done <<< "$ARMS"

echo | tee -a "$LOG"
echo "[$(date -u +%FT%TZ)] ALL-DONE" | tee -a "$LOG"
