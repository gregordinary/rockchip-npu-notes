#!/bin/bash
# PER-SYMBOL IPC ON THE A76s, WHICH IS THE HALF OF EVERY HOST-SIDE CAP THAT IS STILL MISSING.
#
# WHAT IS ALREADY SETTLED. On this arm the A76 INSTRUCTION histogram reads the attention path at
# 39.9% (`expf` 14.07%, `flash_attn_ext_tiled` 12.00%, `fa_mask_scores` 6.38%, `host_softmax_rows`
# 4.02%, `ggml_backend_rocket_flash_attn` 2.96%, `expf@plt` 0.44%), `libggml-rocket.so` at 27.92%
# and `libgomp` at 2.19% [ro-session/trackd24-a76-symbols-12b.md]. The host term is 25.5% of
# prefill wall on this unit [tuning-matrix.md]. An INSTRUCTION share cannot be multiplied by a
# TIME share, so none of those is a cap yet.
#
# WHY THIS IS NOT trackd24 WITH THE EVENT SWAPPED. The deliverable is per-symbol IPC, and taking
# cycles in a SECOND run compares two histograms whose denominators are different runs -- an
# unbudgeted run-to-run term sitting exactly on top of the effect. Both events are recorded HERE
# in ONE capture, so IPC is a within-run ratio and every symbol's two shares share a denominator.
# The A76 PMU has six programmable counters; two events do not multiplex.
#
# THE IDLE TRAP, WHICH WOULD FAKE THE PREDICTED DIRECTION. `perf record -a` samples `swapper`.
# If `cpu_cycles` counts in WFI, every real symbol's CYCLE share falls for a reason that has
# nothing to do with IPC -- and "expf's cycle share is below its instruction share" is precisely
# what this arm predicts. So the run prints `--sort comm` for BOTH events first, and every share
# quoted from it is renormalised over `llama-bench` samples. A swapper share that differs between
# the two events IS the measurement of this trap.
#
# THE CAP THIS PRODUCES, STATED SO IT IS NOT MIS-COMPOSED. `busy_tot` is core-seconds and `phi` is
# a fraction of it, so `H/t = a/phi` is defined such that removing fraction f of host core-seconds
# removes f*(H/t) of the wall. The A76 clock is pinned by governor `performance`, so a symbol's
# share of NON-IDLE A76 cycles IS its share of host core-seconds. The cap for removing a symbol
# entirely is therefore (its share of non-idle A76 cycles) * 0.255 -- NOT its instruction share
# times 0.255, which is the error the 0.7% libgomp cap made.
#
# THE ARM. `gemma4-12b` F16 at `-p 2048 -n 0 -r 3`, PINNED with `taskset 0xf0 -t 4`, governor
# `performance`, NPU at 600 MHz -- the same configuration trackd24 profiled, so the two captures
# are comparable. `taskset` is safe here because this driver invokes the binary directly; it is
# NOT safe in a bench-llm.sh arm's env field.
#
# WHAT A GREEN RESULT WOULD NOT SHOW. Leaf-IP attribution still cannot say who CALLED a symbol,
# and `expf@plt` says at least some of `expf` is reached from `libggml-rocket`. IPC is a property
# of the symbol AS COMPILED HERE -- a vectorised replacement is a different symbol with a
# different IPC, so this bounds what removing the current code buys, not what better code costs.
# One model, one build, one shape. Sampling perturbs the run.
#
# READ `--sort dso` BEFORE ANY SYMBOL VIEW. Every shared object has a symbol at 0x22440.
set -u
OUTD=${OUTD:-$PWD/trackd27-a76ipc12b}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
BIN=${BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
M="$MODELS/gemma4/gemma-4-12b-it-F16.gguf"
FREQ=${FREQ:-299}          # halved against trackd24's 499 because this records TWO events
MASK=${MASK:-0xf0}
mkdir -p "$OUTD"
LOG="$OUTD/a76ipc12b.log"
: > "$LOG"
[ -f "$SO" ] || { echo "MISSING SO: $SO" >&2; exit 2; }
[ -f "$M" ]  || { echo "MISSING MODEL: $M" >&2; exit 2; }

exec >> "$LOG" 2>&1
echo "[$(date -Is)] start  so=$(md5sum "$SO" | cut -d' ' -f1)  mask=$MASK  freq=$FREQ"
echo "--- board ---"
for c in 0 4 6; do echo "  cpu$c gov=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor) min=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_min_freq)"; done
echo "  npu clk=$(sudo -n cat /sys/kernel/debug/clk/clk_summary 2>/dev/null | awk '/scmi_clk_npu/{print $5}')"
echo "--- both events co-exist without multiplexing (want two counts, no <not counted>) ---"
sudo -n perf stat -a -e armv8_cortex_a76/cpu_cycles/,armv8_cortex_a76/inst_retired/ -- sleep 1 2>&1 | tail -8

sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1
echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null 2>&1

echo "[$(date -Is)] warmup"
env GGML_BACKEND_PATH="$SO" taskset "$MASK" "$BIN" -m "$M" -p 512 -n 8 -r 1 -t 4 >/dev/null 2>&1

FIFO="$OUTD/perf.fifo"
rm -f "$FIFO"; mkfifo "$FIFO"
echo "[$(date -Is)] record"
t0=$(date +%s)
sudo -n perf record -a -e armv8_cortex_a76/cpu_cycles/ -e armv8_cortex_a76/inst_retired/ \
     -F "$FREQ" -o "$OUTD/a76.data" -- cat "$FIFO" >/dev/null 2>"$OUTD/perf.err" &
perfpid=$!
sleep 3
if ! kill -0 "$perfpid" 2>/dev/null; then
  echo "  perf record did not start at -F $FREQ; retrying with fixed periods"
  cat "$OUTD/perf.err"
  sudo -n perf record -a -e armv8_cortex_a76/cpu_cycles/ -e armv8_cortex_a76/inst_retired/ \
       -c 4000000 -o "$OUTD/a76.data" -- cat "$FIFO" >/dev/null 2>"$OUTD/perf.err" &
  perfpid=$!
  sleep 3
fi

env GGML_BACKEND_PATH="$SO" ROCKET_LOG_STDERR=1 \
    taskset "$MASK" "$BIN" -m "$M" -p 2048 -n 0 -r 3 -t 4 -o md \
    > "$OUTD/arm.md" 2> "$OUTD/arm.err"
rc=$?
echo end > "$FIFO" 2>/dev/null &
w=0; while [ $w -lt 240 ] && kill -0 "$perfpid" 2>/dev/null; do sleep 0.5; w=$((w+1)); done
kill -0 "$perfpid" 2>/dev/null && sudo -n pkill -x -INT perf >/dev/null 2>&1
wait "$perfpid" 2>/dev/null
t1=$(date +%s)
echo "[$(date -Is)] record rc=$rc wall=$((t1-t0))s  data=$(stat -c %s "$OUTD/a76.data" 2>/dev/null)B"
tail -5 "$OUTD/perf.err"
grep -E "f16-resident|resident" "$OUTD/arm.err" | tail -3
echo "--- ARM t/s (compare against trackd24's 21.19 and the published pinned 21.11) ---"
grep -E "pp2048" "$OUTD/arm.md"

sudo -n chown "$(id -un):$(id -gn)" "$OUTD/a76.data" 2>/dev/null
echo "=== per-event sample counts (perf report prints one section per event) ==="
perf report -i "$OUTD/a76.data" --stdio --sort comm 2>&1 | head -60
echo "=== dso, BOTH events (READ THIS BEFORE ANY SYMBOL VIEW) ==="
perf report -i "$OUTD/a76.data" --stdio --sort dso 2>&1 | head -60
echo "=== dso,symbol top 60, BOTH events ==="
perf report -i "$OUTD/a76.data" --stdio --sort dso,symbol 2>&1 | head -160
echo "=== libgomp only, BOTH events ==="
perf report -i "$OUTD/a76.data" --stdio --sort symbol --dsos libgomp.so.1.0.0 2>&1 | head -40
echo "=== llama-bench comm only (the renormalised denominator), dso,symbol ==="
perf report -i "$OUTD/a76.data" --stdio --sort dso,symbol --comms llama-bench 2>&1 | head -120
rm -f "$FIFO"
echo "[$(date -Is)] TRACKD27-ALL-DONE"
