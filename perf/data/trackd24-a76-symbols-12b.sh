#!/bin/bash
# THE A76 SYMBOL HISTOGRAM, WHICH IS THE HALF OF THE libgomp CAP THAT WAS NEVER MEASURED.
#
# WHAT IS ALREADY SETTLED. `libggml-cpu.so` imports `GOMP_barrier`, `GOMP_parallel` and
# `GOMP_single_start`, so this is an OpenMP build: `ggml_barrier` is `#pragma omp barrier` and
# cannot appear as a leaf symbol, the whole `threadpool->poll` machinery is compiled out, and
# `--poll 0` reached no code [verified on the board, `nm -D`; source-confirmed,
# `ggml/src/ggml-cpu/ggml-cpu.c`]. On the A55s, unpinned, libgomp is **16.02%** of
# `inst_retired` on this model [ro-session/trackd22-a55-symbols-12b.md].
#
# WHY THAT DOES NOT CAP THE TERM. The recorded cap is `16.02% of the little cluster's 4.6% of
# wall` = 0.7%, and both halves leak. The 16.02% is a share of INSTRUCTIONS read as a share of
# TIME, and a spin loop is exactly where those diverge. Worse, the 4.6% is the whole A55
# contribution -- the pin gain -- which pinning removes entirely, while `OMP_WAIT_POLICY` changes
# barrier behaviour on all eight cores. Under the recommended pinned configuration the A55 half is
# zero and every remaining libgomp instruction is on an A76, which no instrument here has ever
# resolved to symbols. So the cap does not enumerate its own term.
#
# THE CAP IS THE OUTPUT OF THIS ARM, NOT AN INPUT TO IT.
#
# THE ARM. `gemma4-12b` F16 at `-p 2048 -n 0 -r 3`, PINNED with `taskset 0xf0 -t 4` so the
# configuration matches the one the recommendation is about, governor `performance`, NPU at
# 600 MHz. `taskset` is safe here because this driver invokes the binary directly; it is NOT safe
# in a bench-llm.sh arm's env field, where `env $envs ... $BIN` eats the assignment after it.
#
# WHAT A GREEN RESULT WOULD NOT SHOW. An instruction share is still not a time share, and
# converting it needs a cycles-based capture this arm does not buy. Leaf-IP attribution cannot say
# who CALLED a symbol. A sampled profile under-reports short leaf functions and a spin loop is a
# long leaf, so the instrument is biased TOWARD finding spin and any spin verdict must be read
# against that bias. Sampling perturbs the run, so the t/s below is not a campaign number. And
# this is one model and one build.
#
# READ `--sort dso` BEFORE ANY SYMBOL VIEW. Every shared object has a symbol at 0x22440, and
# resolving an address by which candidate fits the model is how trackd18's attribution went wrong.
set -u
OUTD=${OUTD:-$PWD/trackd24-a76sym12b}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
BIN=${BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
M="$MODELS/gemma4/gemma-4-12b-it-F16.gguf"
FREQ=${FREQ:-499}
MASK=${MASK:-0xf0}
mkdir -p "$OUTD"
LOG="$OUTD/a76sym12b.log"
: > "$LOG"
[ -f "$SO" ] || { echo "MISSING SO: $SO" >&2; exit 2; }
[ -f "$M" ]  || { echo "MISSING MODEL: $M" >&2; exit 2; }

exec >> "$LOG" 2>&1
echo "[$(date -Is)] start  so=$(md5sum "$SO" | cut -d' ' -f1)  mask=$MASK"

echo "--- OpenMP build check (the whole premise) ---"
for f in "$HOME/npu/llama.cpp/build/bin/libggml-cpu.so"; do
  echo "  $(basename "$f"): $(nm -D "$f" 2>/dev/null | grep -c 'GOMP_\|omp_') GOMP/omp dynamic symbols"
  nm -D "$f" 2>/dev/null | grep -o 'GOMP_[a-z_]*' | sort -u | tr '\n' ' '; echo
done
echo "--- symbol availability ---"
for f in "$BIN" "$SO" "$HOME/npu/llama.cpp/build/bin/libggml-cpu.so" "$HOME/npu/llama.cpp/build/bin/libggml-base.so" /usr/lib/aarch64-linux-gnu/libgomp.so.1; do
  [ -e "$f" ] || { echo "  MISSING $f"; continue; }
  echo "  $(basename "$f"): $(file -b "$f" | grep -o 'not stripped\|stripped' | head -1)  dynsym=$(readelf -sW --dyn-syms "$f" 2>/dev/null | wc -l)  symtab=$(readelf -sW --syms "$f" 2>/dev/null | wc -l)"
done
echo "--- perf event availability ---"
sudo -n perf stat -a -e armv8_cortex_a76/inst_retired/ -- sleep 1 2>&1 | tail -6

sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1
echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null 2>&1

echo "[$(date -Is)] warmup"
env GGML_BACKEND_PATH="$SO" taskset "$MASK" "$BIN" -m "$M" -p 512 -n 8 -r 1 -t 4 >/dev/null 2>&1

FIFO="$OUTD/perf.fifo"
rm -f "$FIFO"; mkfifo "$FIFO"
echo "[$(date -Is)] record"
t0=$(date +%s)
sudo -n perf record -a -e armv8_cortex_a76/inst_retired/ -F "$FREQ" -o "$OUTD/a76.data" \
     -- cat "$FIFO" >/dev/null 2>"$OUTD/perf.err" &
perfpid=$!
sleep 3
if ! kill -0 "$perfpid" 2>/dev/null; then
  echo "  perf record did not start at -F $FREQ; retrying with a fixed period"
  cat "$OUTD/perf.err"
  sudo -n perf record -a -e armv8_cortex_a76/inst_retired/ -c 2000000 -o "$OUTD/a76.data" \
       -- cat "$FIFO" >/dev/null 2>"$OUTD/perf.err" &
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
grep -E "f16-resident|quant-resident|resident" "$OUTD/arm.err" | tail -3
grep -E "pp2048" "$OUTD/arm.md"

sudo -n chown "$(id -un):$(id -gn)" "$OUTD/a76.data" 2>/dev/null
echo "--- perf report: comm ---"
perf report -i "$OUTD/a76.data" --sort comm --stdio 2>&1 | head -25
echo "--- perf report: dso (READ THIS BEFORE ANY SYMBOL VIEW) ---"
perf report -i "$OUTD/a76.data" --sort dso --stdio 2>&1 | head -22
echo "--- perf report: dso,symbol (top 50) ---"
perf report -i "$OUTD/a76.data" --sort dso,symbol --stdio 2>&1 | head -70
echo "--- libgomp only ---"
perf report -i "$OUTD/a76.data" --sort symbol --dsos libgomp.so.1.0.0 --stdio 2>&1 | head -25
rm -f "$FIFO"
echo "[$(date -Is)] TRACKD24-ALL-DONE"
