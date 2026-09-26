#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Same-board whisper-base encoder comparison on the vendor RK1, 20 s window (1000 positions).
# Arms: rknn3 (RKNN whole graph, cores 0-2), rknn1 (core 0), ours (whisper-cli + ggml-rocket
# through rknpu-submit), cpu (whisper-cli alone). Clocks: NPU pinned at 1000 and 600 MHz.
# Protocol matches the mainline RK1's 2026-09-26 re-measure: CPU governor performance on every
# policy, taskset -c 4-7, -t 4, ROCKET_KACC=1, whisper-cli -ac 1000 on jfk.wav, caches dropped
# and memory compacted before every arm, arm order rotated across passes.
#   PW=<sudo password> enc-campaign.sh <outdir> [passes]
set -u
OUT=$1; PASSES=${2:-5}
mkdir -p "$OUT"
E=${E:?set E to the directory holding the .rknn model, librknnrt.so and the harness binaries}
F=${F:?set F to the directory holding whisper.cpp and build-ggml-rocket}
WC=$F/whisper.cpp
SO=$F/build-ggml-rocket/libggml-rocket.so
DF=/sys/class/devfreq/fdab0000.npu
S() { printf '%s\n' "$PW" | sudo -S -p '' "$@"; }
w() { S sh -c 'printf "%s" "$2" > "$1"' _ "$1" "$2"; }   # sysfs write as root; the value is an argument, never stdin

ORIG_NPU_GOV=$(cat $DF/governor); ORIG_NPU_MAX=$(cat $DF/max_freq)
declare -A ORIG_CPU
for p in /sys/devices/system/cpu/cpufreq/policy*; do ORIG_CPU[$p]=$(cat $p/scaling_governor); done
restore() {
    w $DF/max_freq "$ORIG_NPU_MAX"; w $DF/governor "$ORIG_NPU_GOV"
    for p in "${!ORIG_CPU[@]}"; do w $p/scaling_governor "${ORIG_CPU[$p]}"; done
    echo "[$(date -Is)] restored: npu $(cat $DF/governor) max $(cat $DF/max_freq); cpu $(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_governor)" | tee -a "$OUT/campaign.log"
}
trap restore EXIT
trap 'exit 1' INT TERM HUP   # route a signal through the EXIT trap, so the board is never left pinned

for p in "${!ORIG_CPU[@]}"; do
    w $p/scaling_governor performance
    [ "$(cat $p/scaling_governor)" = performance ] || { echo "ABORT: $p governor did not take" | tee -a "$OUT/campaign.log"; exit 1; }
done
set_npu() {  # $1 Hz; aborts unless the clock reads back pinned
    w $DF/governor performance
    w $DF/max_freq "$1"; w $DF/min_freq 300000000
    sleep 1
    [ "$(cat $DF/governor)" = performance ] && [ "$(cat $DF/cur_freq)" = "$1" ] \
        || { echo "ABORT: npu pin to $1 read back $(cat $DF/governor) $(cat $DF/cur_freq)" | tee -a "$OUT/campaign.log"; exit 1; }
}
reset_mem() { sync; w /proc/sys/vm/drop_caches 3; w /proc/sys/vm/compact_memory 1; sleep 2; }
temp() { awk '{s = s " " int($1/1000)} END {print s}' /sys/class/thermal/thermal_zone*/temp; }

run_arm() {  # $1 arm, $2 clock label, $3 pass
    local tag="p$3-$2-$1" log="$OUT/p$3-$2-$1.log" v
    reset_mem
    local f0; f0=$(cat $DF/cur_freq)
    case $1 in
        rknn3) taskset -c 4-7 $E/rknn_enc_bench $E/whisper_encoder_base_20s.rknn 012 5 20 > "$log" 2>&1
               v=$(awk '/^RESULT/{print $5 " setrunget=" $11}' "$log");;
        rknn1) taskset -c 4-7 $E/rknn_enc_bench $E/whisper_encoder_base_20s.rknn 0 3 10 > "$log" 2>&1
               v=$(awk '/^RESULT/{print $5 " setrunget=" $11}' "$log");;
        ours)  GGML_BACKEND_PATH=$SO ROCKET_KACC=1 taskset -c 4-7 $WC/build/bin/whisper-cli \
                   -m $WC/models/ggml-base.en.bin -f $WC/samples/jfk.wav -ac 1000 -t 4 > "$log" 2>&1
               v=$(grep -oE 'encode time = +[0-9.]+' "$log" | awk '{print $4}')
               grep -q 'using ROCKET backend' "$log" || v="$v NO-ROCKET";;
        cpu)   taskset -c 4-7 $WC/build/bin/whisper-cli \
                   -m $WC/models/ggml-base.en.bin -f $WC/samples/jfk.wav -ac 1000 -t 4 > "$log" 2>&1
               v=$(grep -oE 'encode time = +[0-9.]+' "$log" | awk '{print $4}');;
    esac
    printf '%s\t%s\t%s\t%s\tnpu_hz=%s->%s\ttemp=%s\n' "$3" "$2" "$1" "${v:-FAIL}" "$f0" "$(cat $DF/cur_freq)" "$(temp)" \
        | tee -a "$OUT/results.tsv"
}

echo "[$(date -Is)] start: $PASSES passes; cpu gov $(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_governor); temps$(temp)" | tee "$OUT/campaign.log"
ARMS=(rknn3 rknn1 ours cpu)
for ((p = 1; p <= PASSES; p++)); do
    if ((p % 2)); then CLOCKS=(1000 600); else CLOCKS=(600 1000); fi
    for c in "${CLOCKS[@]}"; do
        set_npu "${c}000000"
        echo "[$(date -Is)] pass $p clock $c: governor $(cat $DF/governor) cur $(cat $DF/cur_freq)" | tee -a "$OUT/campaign.log"
        n=${#ARMS[@]}
        for ((i = 0; i < n; i++)); do run_arm "${ARMS[$(((i + p) % n))]}" "$c" "$p"; done
    done
done
echo "[$(date -Is)] CAMPAIGN-DONE" | tee -a "$OUT/campaign.log"
