#!/bin/bash
# IS THE NON-MONOTONIC fd CURVE REAL, OR ONE PROCESS EACH? The repeat pass.
#
# The first ladder read 286 at 5 workers (twice), 293 at 6, 274 at 7, at near-identical resident
# budgets (21028 / 21186 / 21223 MB). That is a peak at 6 and a value at 7 BELOW the default, and
# it refutes a recorded claim that the fd fix alone moves placement by zero weights. The claim it
# refutes was tested at one worker count only, so before replacing it the two new points need a
# second reading of their own.
#
# Placement has been deterministic per configuration everywhere it has been repeated in this
# workspace, so a disagreement here would itself be the finding.
#
# PLACEMENT ONLY, -p 512, same shape as the first ladder. No cap is quoted: this reads which limit
# fires and how many weights land, not what a weight is worth.
set -u
export MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
export OUTD=$PWD/trackd19b-f16window-repeat
export PROMPT="-p 512 -n 0 -r 1"
export SO=$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so
export BIN=$HOME/npu/llama.cpp/build/bin/llama-bench
export ARMS='t6_b|ROCKET_N_THREADS=6
t7_b|ROCKET_N_THREADS=7
t6_c|ROCKET_N_THREADS=6'
mkdir -p "$OUTD"
bash "$(dirname "$0")"/trackd13-iova-readout.sh > "$OUTD/driver.log" 2>&1
echo "[$(date -Is)] LAUNCHER-DONE rc=$?" >> "$OUTD/driver.log"
# The chain follows immediately so the board is never idle.
bash "$(dirname "$0")"/chain2.sh
