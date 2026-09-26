#!/bin/bash
# WHICH ADMISSION LIMIT STOPS THE 12B AT 87% RESIDENT -- a readout, not a campaign.
#
# `ROCKET_F16_RESIDENT=auto` on gemma-4-12b-it-F16 places 286 of 328 weights (18078MB) and
# streams the other 42. THREE limits can turn a weight away, and `build_resident` checks them
# in this order: the byte budget, the runtime RAM floor, then the pack itself (IOVA). Each
# takes a DIFFERENT fix -- a larger budget, more free RAM, more worker fds -- so the placed
# fraction alone cannot be acted on. `[f16-resident] admission first declined` names the one
# that fired first, and that line is the whole measurement here.
#
# WHAT THIS SETTLES. The open-work list records the ceiling as the NPU IOVA window, citing
# `ROCKET_CREATE_BO: No space left on device` and hinting ROCKET_N_THREADS for more fds. All
# four resident arms of the trackd12 2x2 instead report the RESERVE FLOOR, MemAvailable
# 9479-9486MB against a 9535MB floor, at the same 18078MB. Those take opposite fixes, so the
# arms below apply each fix and read which one moves placement.
#
# THE RESERVE IS ONE KNOB DOING TWO JOBS. reserve = max(6GiB, 30% of MemTotal) = 9535MB on
# this 31785MB board, and the same value BOTH sizes the budget (MemAvailable - reserve) AND
# arms the runtime latch. So lowering it raises the budget and lowers the floor together, and
# a move under it does not say which of the two was binding.
#
# THIS BOARD HAS NO SWAP AND dmesg ALREADY CARRIES THREE OOM KILLS. An over-commit is a hard
# kill, which is why the reserve is deliberately generous. The low-reserve arms run LAST, each
# arm checks dmesg for a kill it caused, and a killed arm is VOID rather than a data point.
# Do not lower the reserve below the built-in 6144MB floor.
#
# This measures PLACEMENT ONLY. It does not measure what closing the 42 streamed weights buys:
# the per-call pack's share of this model's wall is unmeasured, so no cap is quoted here.
set -u
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/gemma4/gemma-4-12b-it-F16.gguf"
SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
BIN=${BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
OUTD=${OUTD:-$MODELS/trackd-iova}
PROMPT=${PROMPT:--p 512 -n 0 -r 1}
LOG="$OUTD/readout.md"

# label|extra env  -- the baseline is ROCKET_F16_RESIDENT=auto at DEFAULT threads and reserve,
# not the campaigns' 'stock' (which means no residency knob at all). Then the fd fix, then the
# RAM fix stepping down.
ARMS=${ARMS:-'auto_default|
nthreads8|ROCKET_N_THREADS=8
reserve7168|ROCKET_QUANT_RESIDENT_RESERVE_MB=7168
reserve6144|ROCKET_QUANT_RESIDENT_RESERVE_MB=6144'}

mkdir -p "$OUTD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "MISSING: $BIN" >&2; exit 2; }
[ -f "$SO" ]    || { echo "MISSING: $SO" >&2; exit 2; }

oom_count() { sudo dmesg 2>/dev/null | grep -icE "out of memory|oom-kill|oom_reaper"; }
OOM0=$(oom_count)

{
  echo "<!-- gemma4-12b F16 residency admission readout   PROMPT='$PROMPT'"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     so=$(md5sum "$SO" | cut -d' ' -f1)"
  echo "     MemTotal=$(awk '/MemTotal/{print $2}' /proc/meminfo) kB  SwapTotal=$(awk '/SwapTotal/{print $2}' /proc/meminfo) kB"
  echo "     dmesg OOM lines before start: $OOM0"
  echo "-->"
  echo
} > "$LOG"

echo "[$(date -u +%FT%TZ)] readout start, $(echo "$ARMS" | wc -l) arms" | tee -a "$LOG"

while IFS='|' read -r label envs; do
  [ -n "$label" ] || continue
  err="$OUTD/$label.err"
  echo | tee -a "$LOG"
  echo "### $label  env='$envs'  $(date -u +%T)" | tee -a "$LOG"

  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null 2>&1
  avail_before=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  echo "    MemAvailable before: $((avail_before/1024)) MB" | tee -a "$LOG"

  # shellcheck disable=SC2086
  env GGML_BACKEND_PATH="$SO" ROCKET_LOG_STDERR=1 ROCKET_F16_RESIDENT=auto $envs \
      "$BIN" -m "$MODEL" $PROMPT -o md >/dev/null 2>"$err"
  rc=$?
  echo "    rc=$rc" | tee -a "$LOG"

  grep -h "\[rocket\].*resident budget"      "$err" | sed 's/^ *//;s/^/    /' | tee -a "$LOG"
  grep -h "f16-resident"                     "$err" | sed 's/^ *//;s/^/    /' | tee -a "$LOG"
  grep -hiE "No space left|CREATE_BO"        "$err" | head -3 | sed 's/^ *//;s/^/    ALLOC: /' | tee -a "$LOG"

  oom_now=$(oom_count)
  if [ "$oom_now" -gt "$OOM0" ]; then
    echo "    *** OOM KILL DETECTED during this arm -- VOID, and stopping the readout ***" | tee -a "$LOG"
    sudo dmesg | grep -iE "out of memory|oom-kill" | tail -3 | sed 's/^/    /' | tee -a "$LOG"
    break
  fi
  [ "$rc" -eq 0 ] || echo "    (non-zero rc -- read the stderr at $err before trusting this arm)" | tee -a "$LOG"
done <<< "$ARMS"

echo | tee -a "$LOG"
echo "[$(date -u +%FT%TZ)] ALL-DONE" | tee -a "$LOG"
echo "stderr kept under $OUTD"
