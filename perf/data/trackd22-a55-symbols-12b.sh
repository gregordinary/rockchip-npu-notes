#!/bin/bash
# THE A55 SYMBOL HISTOGRAM ON A PLAIN-TRANSFORMER MODEL. The out-of-sample half of trackd18.
#
# WHAT trackd18 SETTLED, so this does not re-ask it. On `qwen35-9b` the little cluster's retired
# instructions are real graph work and not barrier spin: 97524 samples, none lost, 97.20% inside
# `llama-bench`, no `ggml_barrier` anywhere, and `librocketnpu`'s own host symbols total 0.43%
# [HW readout 2026-09-02, RK1; raw in ro-session/trackd18-a55-symbols.md].
#
# WHY THAT ANSWER IS NARROW. Its two largest entries -- `gated_delta_net` 20.79% and `ssm_conv`
# 17.31% -- are that model's hybrid-attention layers, which have no NPU handler at all. On a model
# built from them the answer is close to forced: work with no handler stays on the CPU, and enough
# of it lands on the little cluster to fill the histogram. It says much less about a PLAIN
# transformer, where every large op has a handler and what is left is glue.
#
# THE ARM. `gemma4-12b` F16, spelled exactly as the published `unpinned` arm of the pinning
# intervention: `GGML_BACKEND_PATH` and nothing else, no extra llama-bench args, `-p 2048 -n 0
# -r 3`. That arm's recorded `a55_inst_share` is **0.141** and its `pin76` gain is **1.046x**
# [perf/data/tuning-matrix.md], so the histogram decomposes a number that is already published and
# already load-bearing -- it is the low end of the three-model share-versus-gain ordering. Note
# that this arm carries NO `ROCKET_KACC`, because bench-llm.sh's ARMS mode does not add it and the
# published row was taken that way; trackd18's arm did carry it, so the two are not a matched pair
# in that respect.
#
# The model streams every weight in this configuration (0 placed), so the host's per-call weight
# pack runs on every call -- and the pack is pinned to the big cores [source-confirmed,
# `rocket_affinity.c`], which is the reason to expect the driver's own symbols to stay small here
# as they did on the 9B rather than to grow with the streaming.
#
# NOT A PERF LEVER, SO NO CAP APPLIES. Nothing here proposes to make anything faster.
#
# WHAT A GREEN RESULT WOULD NOT SHOW. A symbol histogram is a PROPORTION, not a count -- it
# cannot say how much wall the term is worth, only how the little cluster's instructions divide.
# Leaf-IP attribution cannot say who CALLED a symbol. A sampled profile under-reports short leaf
# functions and a spin loop is a long leaf, so the instrument is biased TOWARD finding spin and a
# spin verdict must be read against that bias. Sampling perturbs the run, so the t/s printed below
# is not comparable to any campaign number. And this is one more model, not a class.
set -u
OUTD=${OUTD:-$PWD/trackd22-a55sym12b}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
BIN=${BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
M="$MODELS/gemma4/gemma-4-12b-it-F16.gguf"
FREQ=${FREQ:-499}
mkdir -p "$OUTD"
LOG="$OUTD/a55sym12b.log"
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

# Warm-up OUTSIDE the capture: spins the NPU clock off idle and pays the model load, so the
# recorded region is steady-state prefill.
echo "[$(date -Is)] warmup"
env GGML_BACKEND_PATH="$SO" "$BIN" -m "$M" -p 512 -n 8 -r 1 >/dev/null 2>&1

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

env GGML_BACKEND_PATH="$SO" ROCKET_LOG_STDERR=1 \
    "$BIN" -m "$M" -p 2048 -n 0 -r 3 -o md \
    > "$OUTD/arm.md" 2> "$OUTD/arm.err"
rc=$?
echo end > "$FIFO" 2>/dev/null &
w=0; while [ $w -lt 240 ] && kill -0 "$perfpid" 2>/dev/null; do sleep 0.5; w=$((w+1)); done
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
echo "[$(date -Is)] TRACKD22-ALL-DONE"
