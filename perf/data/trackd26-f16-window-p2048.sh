#!/bin/bash
# DOES SIX WORKERS BUY PLACEMENT ON THE DENSE f16 ROUTE AT THE CAMPAIGN SHAPE?
#
# WHY THIS EXISTS. Six is the recommended `ROCKET_N_THREADS` on both routes, and on the dense f16
# route the only measurement behind it is PLACEMENT at `-p 512`: 293 weights at six against 286 at
# five, with the route first-declining on the NPU IOVA window and a real `ROCKET_CREATE_BO` ENOSPC
# beside it [ro-session/trackd19-f16-window-ladder.md]. Its wall was never measured and cannot be:
# the inferred cap is 0.15% of prefill, below this instrument's resolution. **The result is the
# cap**, so a wall campaign is the mistake and this is a readout.
#
# WHAT IS ACTUALLY UNDER TEST. Which limit binds is a function of the SHAPE, not of the route:
# `-p 512` at five workers announces the IOVA window, and the 2026-09-01 pp2048 campaign read this
# same unit as RAM-bound, because the KV cache lowers MemAvailable and the reserve floor catches
# first. A worker count buys fds and fds relieve the WINDOW; nothing about `ROCKET_N_THREADS`
# moves a memory floor. So the `-p 512` ladder may not transfer to the shape every campaign number
# in the matrix is taken at.
#
# THE PREDICTION, REGISTERED BEFORE THE RUN, 2026-09-02: both arms report the
# reserve floor and their placed counts differ by 0-2 of 328. RIVAL: the window still binds, `t5`
# announces it, and `t6` places 5-10 more, reproducing the `-p 512` ladder.
#
# A NULL IS THE FINDING, NOT A PASS. If placement saturates identically, the recommendation of six
# on this route rests on the `-p 512` ladder alone and should be recorded that way.
#
# t5 IS RUN TWICE because a first-decline reason string is a race when two limits nearly coincide,
# and on this unit at `-p 512` they do -- at nw=8 the floor caught at the same 286 the window
# stopped at. Reading one arm cannot tell a stable reason from a coin flip.
#
# PLACEMENT ONLY, no timed run, no cap quoted.
set -u
export MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
export OUTD=${OUTD:-$PWD/trackd26-f16window-p2048}
export PROMPT="-p 2048 -n 0 -r 1"
export SO=$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so
export BIN=$HOME/npu/llama.cpp/build/bin/llama-bench
export ARMS='t5_a|
t5_b|
t6_a|ROCKET_N_THREADS=6
t6_b|ROCKET_N_THREADS=6'
mkdir -p "$OUTD"
bash "$(dirname "$0")"/trackd13-iova-readout.sh > "$OUTD/driver.log" 2>&1
echo "[$(date -Is)] TRACKD26-LAUNCHER-DONE rc=$?" >> "$OUTD/driver.log"
