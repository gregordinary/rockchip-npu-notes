#!/bin/bash
# LOCALIZING THE MOVING FA CHECKSUM.
#
# WHAT IS MEASURED. Five runs on one build, one config, gave 8852206bf3ccda62 four times and
# 345a87c827fa972b once, and the odd run was a k=1 run with two k=1 runs agreeing beside it. So
# the handler's output is not reproducible run to run, the rate is about 1 in 5 on this unit, and
# it is NOT a function of ROCKET_FA_THREADS. That retires the aggregate hash as a gate on the
# split: it compares one run to one run, and its two arms disagree at the same rate whether or
# not they differ in k.
#
# WHAT THIS ASKS. ROCKET_FA_CHECKSUM=2 prints a hash per op with its shape. Six runs at ONE
# setting diff to the op index, which separates the two candidates a single number cannot:
#   * ALWAYS THE SAME op index / n_kv / head count -> shape-selective, a wrong branch on one
#     geometry, and the next step is that geometry's own gate.
#   * A DIFFERENT op each time -> a race or an uninitialized read, and the next step is the
#     driver's worker path rather than a shape.
# Six runs at one setting also measure the rate better than five split across two settings did.
#
# THE OP COUNT IS THE FIRST THING TO READ. 288 in every run so far; a run with a different count
# ran a different graph and its diff is not a diff of the same thing.
set -u
OUTD=${OUTD:-$PWD/trackd38-cksum-localize}
MODEL=${MODEL:-${MODELS:?set MODELS to the directory holding the GGUFs}/gemma4/gemma-4-12b-it-F16.gguf}
BIN=$HOME/npu/llama.cpp/build/bin/llama-bench
SO=$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so
LOG="$OUTD/localize.log"
mkdir -p "$OUTD"
{
  echo "[$(date -Is)] FA per-op checksum, six runs at ROCKET_FA_THREADS=1"
  echo "  so=$(md5sum "$SO" | cut -d' ' -f1)"
  echo "  librocketnpu.a=$(md5sum /usr/local/lib/librocketnpu.a | cut -d' ' -f1)"
  for run in 1 2 3 4 5 6; do
    echo "--- run $run ---"
    sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    env GGML_BACKEND_PATH="$SO" ROCKET_KACC=1 ROCKET_FA_THREADS=1 \
      ROCKET_FA_CHECKSUM=2 ROCKET_FA_TIMING=1 ROCKET_LOG_STDERR=1 \
      taskset 0xf0 "$BIN" -m "$MODEL" -p 2048 -n 0 -r 1 -t 4 -v \
      > "$OUTD/run$run.out" 2> "$OUTD/run$run.err"
    echo "  rc=$?"
    grep -a "ROCKET FA checksum" "$OUTD/run$run.err" | tail -1 || echo "  NO CHECKSUM LINE"
    grep -a "ROCKET FA op " "$OUTD/run$run.err" | sed 's/.*ROCKET FA op /op /' > "$OUTD/ops$run.txt"
    echo "  per-op lines: $(wc -l < "$OUTD/ops$run.txt")"
  done
  echo "--- pairwise diffs against run 1 ---"
  for run in 2 3 4 5 6; do
    n=$(diff "$OUTD/ops1.txt" "$OUTD/ops$run.txt" | grep -c '^<' || true)
    echo "  run1 vs run$run: $n differing op lines"
    diff "$OUTD/ops1.txt" "$OUTD/ops$run.txt" | head -8
  done
  echo "[$(date -Is)] LOCALIZE-DONE"
} >> "$LOG" 2>&1
