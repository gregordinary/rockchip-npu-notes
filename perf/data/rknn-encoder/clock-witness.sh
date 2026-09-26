#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Clock witness + attribution for the RKNN encoder: 300/600/1000 MHz, timed arm then perf-detail arm.
#   PW=<sudo password> clock-witness.sh <outdir>
set -u
OUT=$1; mkdir -p "$OUT"
E=${E:?set E to the directory holding the .rknn model, librknnrt.so and the harness binaries}; M=$E/whisper_encoder_base_20s.rknn
DF=/sys/class/devfreq/fdab0000.npu
S() { printf '%s\n' "$PW" | sudo -S -p '' "$@"; }
w() { S sh -c 'printf "%s" "$2" > "$1"' _ "$1" "$2"; }
ORIG_NPU_GOV=$(cat $DF/governor); ORIG_NPU_MAX=$(cat $DF/max_freq)
declare -A ORIG_CPU
for p in /sys/devices/system/cpu/cpufreq/policy*; do ORIG_CPU[$p]=$(cat $p/scaling_governor); done
restore() { w $DF/max_freq "$ORIG_NPU_MAX"; w $DF/governor "$ORIG_NPU_GOV"
            for p in "${!ORIG_CPU[@]}"; do w $p/scaling_governor "${ORIG_CPU[$p]}"; done
            echo "restored: npu $(cat $DF/governor) cpu $(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_governor)"; }
trap restore EXIT; trap 'exit 1' INT TERM HUP
for p in "${!ORIG_CPU[@]}"; do w $p/scaling_governor performance; done
for c in 1000 300 600 1000; do
    w $DF/governor performance; w $DF/max_freq "${c}000000"; w $DF/min_freq 300000000; sleep 1
    [ "$(cat $DF/cur_freq)" = "${c}000000" ] || { echo "ABORT pin $c read $(cat $DF/cur_freq)"; exit 1; }
    for mask in 0 012; do
        taskset -c 4-7 $E/rknn_enc_bench $M $mask 3 10 > "$OUT/bench-$c-$mask.log" 2>&1
        taskset -c 4-7 $E/rknn_enc_perf $M $mask 2 > "$OUT/perf-$c-$mask.log" 2>&1
        printf '%s MHz mask %-3s bench %s | perf_run_us %s\n' $c $mask \
            "$(awk '/^RESULT/{print $5}' "$OUT/bench-$c-$mask.log")" \
            "$(awk '/^PERF_RUN/{v=$NF} END{print v}' "$OUT/perf-$c-$mask.log")"
    done
done
