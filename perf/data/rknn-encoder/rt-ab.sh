#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# librknnrt 2.3.2 vs 2.3.0 on the same model, NPU pinned at 1000 MHz, CPU governor performance.
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
w $DF/governor performance; w $DF/max_freq 1000000000; sleep 1
[ "$(cat $DF/cur_freq)" = 1000000000 ] || { echo "ABORT pin read $(cat $DF/cur_freq)"; exit 1; }
for rep in 1 2; do
    for rt in 232 230; do
        d=$E; [ $rt = 230 ] && d=$E/rt230
        v=$(taskset -c 4-7 $d/rknn_enc_bench $M 012 5 20 2>&1 | tee "$OUT/bench-$rt-$rep.log" | awk '/^api/{a=$2} /^RESULT/{print a, $5}')
        echo "rep $rep rt $rt 3-core rknn_run median: $v"
    done
done
for rt in 232 230; do
    d=$E; [ $rt = 230 ] && d=$E/rt230
    taskset -c 4-7 $d/rknn_enc_perf $M 012 2 > "$OUT/perf-$rt.log" 2>&1
    echo "rt $rt perf: exSDPAttention $(awk '$1=="exSDPAttention" && NF>=6 {print $(NF-2)}' "$OUT/perf-$rt.log" | tail -1) us, total $(awk '/^Total  /{print $NF}' "$OUT/perf-$rt.log") us"
done
