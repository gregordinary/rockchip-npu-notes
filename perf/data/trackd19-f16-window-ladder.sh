#!/bin/bash
# WHAT DOES RAISING ROCKET_N_THREADS BUY WHERE THE IOVA WINDOW IS THE ANNOUNCED BINDER?
#
# trackd13 established that on gemma4-12b F16 at -p 512 the f16 residency route first-declines
# on the WINDOW, not the floor: three different RAM budgets (21095 / 23466 / 24494 MB) all place
# the identical 286 weights / 18078 MB and all announce "the NPU IOVA window filled", with a real
# ROCKET_CREATE_BO ENOSPC beside them. That is the case `API.md`'s ROCKET_N_THREADS prescription
# is written for, and this reads what the prescription actually yields there.
#
# The knob's own ladder was never run: trackd13 tested only 5 and 8, and trackd13b only the joint
# 8 + reserve-6144 cell. So 6 and 7 are unmeasured, and whether ANY worker count buys placement
# on this route at this shape is unmeasured with it.
#
# THE CONTROL THAT MUST SUCCEED is t8_r6144: trackd13b read 328 of 328 / 100% resident twice on
# that cell. If it does not reproduce here, nothing else in this readout can be read either.
#
# t5 IS RUN TWICE because a first-decline reason string is a race when two limits nearly coincide,
# and on this unit they do -- at nw=8 the floor catches at the SAME 286 the window stopped at.
#
# PLACEMENT ONLY, and no cap is quoted: this reads which limit fires and how many weights land,
# not what a weight is worth. -p 512 is deliberate and is NOT the campaign shape -- at -p 2048 the
# KV cache lowers MemAvailable and the RESERVE FLOOR becomes the binder instead, which is why the
# 2026-09-01 pp2048 campaign read this unit as RAM-bound. Which limit binds is a function of the
# SHAPE.
set -u
export MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
export OUTD=$PWD/trackd19-f16window
export PROMPT="-p 512 -n 0 -r 1"
export SO=$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so
export BIN=$HOME/npu/llama.cpp/build/bin/llama-bench
export ARMS='t5_a|
t5_b|
t6|ROCKET_N_THREADS=6
t7|ROCKET_N_THREADS=7
t8|ROCKET_N_THREADS=8
t8_r6144|ROCKET_N_THREADS=8 ROCKET_QUANT_RESIDENT_RESERVE_MB=6144'
mkdir -p "$OUTD"
bash "$(dirname "$0")"/trackd13-iova-readout.sh > "$OUTD/driver.log" 2>&1
echo "[$(date -Is)] LAUNCHER-DONE rc=$?" >> "$OUTD/driver.log"
