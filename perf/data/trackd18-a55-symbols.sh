#!/bin/bash
# WHAT ARE THE A55 INSTRUCTIONS? The per-thread instrument for the little-cluster share.
#
# WHAT IS ALREADY SETTLED, so this does not re-derive it. `librocketnpu` pins every worker to a
# big core [source-confirmed, `rocket_affinity.c`], so the little-cluster instructions the
# readout's `a55_inst_share` column counts belong to llama.cpp's own threads, not to the driver's.
# `ggml_barrier` spins unconditionally with no poll budget and no sleep fallback
# [source-confirmed, `ggml/src/ggml-cpu/ggml-cpu.c`], and `--poll` does not reach it. What is NOT
# settled is whether those instructions are that barrier spin or real graph work -- which decides
# whether anything but cores can reach the term `taskset 0xf0` removes.
#
# THE INSTRUMENT. `perf record -a -e armv8_cortex_a55/inst_retired/` over ONE unpinned arm, then
# `perf report --sort comm,symbol`. LEAF-IP ATTRIBUTION ONLY: no call graph, so it needs neither
# frame pointers nor unwinding, and it cannot say who CALLED a symbol. The event is A55-specific,
# so perf opens it only on the four little cores and the capture is bounded to them.
#
# PERF WRAPS A FIFO, NOT THE WORKLOAD. `perf record -a -- llama-bench` would run the model as
# ROOT, which is a different process than every other arm in this workspace. So perf's workload is
# a `cat` blocked on a fifo, the model runs as the normal user beside it, and the region ends with
# a WRITE rather than a signal -- backgrounding `sudo perf` gives you sudo's pid, so a SIGINT lands
# on sudo and perf never sees it. This is the pattern bench-llm.sh's readout already uses.
#
# NOT A PERF LEVER, SO NO CAP APPLIES. Nothing here proposes to make anything faster.
#
# WHAT A GREEN RESULT WOULD NOT SHOW. A symbol histogram is a PROPORTION, not a count -- it
# cannot say how much wall the term is worth, only how the little cluster's instructions divide.
# A sampled profile under-reports short leaf functions, and a spin loop is a long leaf, so the
# instrument is biased TOWARD finding spin. Read a spin verdict against that bias. Sampling also
# perturbs the run, so the t/s printed below is not comparable to any campaign number.
#
# THE ARM is the unit the pinning interaction was measured on: qwen35-9b Q4_K_M under
# `ROCKET_QUANT_RESIDENT=auto -b 2048 -ub 2048`, unpinned, whose published `a55_inst_share` of
# 0.1419 gives the number this histogram must decompose.
set -u
OUTD=${OUTD:-$PWD/trackd18-a55sym}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
BIN=${BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
M="$MODELS/qwen35/Qwen3.5-9B-Q4_K_M.gguf"
FREQ=${FREQ:-499}
mkdir -p "$OUTD"
LOG="$OUTD/a55sym.log"
: > "$LOG"
[ -f "$SO" ] || { echo "MISSING SO: $SO" >&2; exit 2; }
[ -f "$M" ]  || { echo "MISSING MODEL: $M" >&2; exit 2; }

exec >> "$LOG" 2>&1
echo "[$(date -Is)] start  so=$(md5sum "$SO" | cut -d' ' -f1)"

# Are the binaries symbolised at all? A stripped .so turns every row into a raw address and the
# histogram says nothing -- read this BEFORE the capture rather than discovering it in the report.
echo "--- symbol availability ---"
for f in "$BIN" "$SO" "$HOME/npu/llama.cpp/build/bin/libggml-cpu.so" "$HOME/npu/llama.cpp/build/bin/libggml-base.so"; do
  [ -e "$f" ] || { echo "  MISSING $f"; continue; }
  echo "  $(basename "$f"): $(file -b "$f" | grep -o 'not stripped\|stripped' | head -1)  dynsym=$(readelf -sW --dyn-syms "$f" 2>/dev/null | wc -l)  symtab=$(readelf -sW --syms "$f" 2>/dev/null | wc -l)"
done
echo "--- perf event availability ---"
sudo -n perf stat -a -e armv8_cortex_a55/inst_retired/ -- sleep 1 2>&1 | tail -6

sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1
echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null 2>&1

# Warm-up OUTSIDE the capture: pays the residency ingest and spins the NPU clock off idle, so the
# recorded region is steady-state prefill and not a one-time load.
echo "[$(date -Is)] warmup"
env GGML_BACKEND_PATH="$SO" ROCKET_KACC=1 ROCKET_QUANT_RESIDENT=auto \
    "$BIN" -m "$M" -p 512 -n 8 -r 1 -b 2048 -ub 2048 >/dev/null 2>&1

FIFO="$OUTD/perf.fifo"
rm -f "$FIFO"; mkfifo "$FIFO"
echo "[$(date -Is)] record"
t0=$(date +%s)
sudo -n perf record -a -e armv8_cortex_a55/inst_retired/ -F "$FREQ" -o "$OUTD/a55.data" \
     -- cat "$FIFO" >/dev/null 2>"$OUTD/perf.err" &
perfpid=$!
sleep 3
if ! kill -0 "$perfpid" 2>/dev/null; then
  echo "  perf record did not start at -F $FREQ; retrying with a fixed period"
  cat "$OUTD/perf.err"
  sudo -n perf record -a -e armv8_cortex_a55/inst_retired/ -c 2000000 -o "$OUTD/a55.data" \
       -- cat "$FIFO" >/dev/null 2>"$OUTD/perf.err" &
  perfpid=$!
  sleep 3
fi

env GGML_BACKEND_PATH="$SO" ROCKET_KACC=1 ROCKET_QUANT_RESIDENT=auto ROCKET_LOG_STDERR=1 \
    "$BIN" -m "$M" -p 2048 -n 0 -r 3 -b 2048 -ub 2048 -o md \
    > "$OUTD/arm.md" 2> "$OUTD/arm.err"
rc=$?
echo end > "$FIFO" 2>/dev/null &
w=0; while [ $w -lt 120 ] && kill -0 "$perfpid" 2>/dev/null; do sleep 0.5; w=$((w+1)); done
kill -0 "$perfpid" 2>/dev/null && sudo -n pkill -x -INT perf >/dev/null 2>&1
wait "$perfpid" 2>/dev/null
t1=$(date +%s)
echo "[$(date -Is)] record rc=$rc wall=$((t1-t0))s  data=$(stat -c %s "$OUTD/a55.data" 2>/dev/null)B"
cat "$OUTD/perf.err" | tail -5
grep -E "f16-resident|quant-resident|resident" "$OUTD/arm.err" | tail -3
grep -E "pp2048" "$OUTD/arm.md"

# Hand the capture to the invoking user and read it AS that user. `perf report` refuses a file
# that is owned by neither the current user nor root, so chowning to debian and then reading
# under sudo fails with "not owned by current user or root" and prints an EMPTY report -- which
# reads exactly like a capture with no samples. The sample count on the record line above is what
# says the capture worked; check it before believing an empty histogram.
sudo -n chown "$(id -un):$(id -gn)" "$OUTD/a55.data" 2>/dev/null
echo "--- perf report: comm ---"
perf report -i "$OUTD/a55.data" --sort comm --stdio 2>&1 | head -25
echo "--- perf report: comm,symbol (top 45) ---"
perf report -i "$OUTD/a55.data" --sort comm,symbol --stdio 2>&1 | head -60
echo "--- perf report: dso ---"
perf report -i "$OUTD/a55.data" --sort dso --stdio 2>&1 | head -20
rm -f "$FIFO"
echo "[$(date -Is)] ALL-DONE"
