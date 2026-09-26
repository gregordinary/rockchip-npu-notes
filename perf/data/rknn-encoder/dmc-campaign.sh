#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# DDR devfreq A/B on the vendor RK1: dmc_ondemand (od) against dmc pinned at 2112 MHz (pin).
# CPU governor performance, taskset -c 4-7, caches dropped before every arm, DDR frequency
# sampled every 50 ms during each arm.
#   PW=<sudo password> dmc-campaign.sh <outdir> [passes]
set -u
OUT=$1; PASSES=${2:-3}; mkdir -p "$OUT"
E=${E:?set E to the directory holding the .rknn model, librknnrt.so and the harness binaries}; M=$E/whisper_encoder_base_20s.rknn
F=${F:?set F to the directory holding whisper.cpp and build-ggml-rocket}; WC=$F/whisper.cpp; SO=$F/build-ggml-rocket/libggml-rocket.so
DF=/sys/class/devfreq/fdab0000.npu; DM=/sys/class/devfreq/dmc
S() { printf '%s\n' "$PW" | sudo -S -p '' "$@"; }
w() { S sh -c 'printf "%s" "$2" > "$1"' _ "$1" "$2"; }
ORIG_NPU_GOV=$(cat $DF/governor); ORIG_NPU_MAX=$(cat $DF/max_freq); ORIG_DMC_GOV=$(cat $DM/governor)
declare -A ORIG_CPU
for p in /sys/devices/system/cpu/cpufreq/policy*; do ORIG_CPU[$p]=$(cat $p/scaling_governor); done
restore() {
    w $DF/max_freq "$ORIG_NPU_MAX"; w $DF/governor "$ORIG_NPU_GOV"; w $DM/governor "$ORIG_DMC_GOV"
    for p in "${!ORIG_CPU[@]}"; do w $p/scaling_governor "${ORIG_CPU[$p]}"; done
    echo "[$(date -Is)] restored: npu $(cat $DF/governor) dmc $(cat $DM/governor) cpu $(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_governor)" | tee -a "$OUT/campaign.log"
}
trap restore EXIT; trap 'exit 1' INT TERM HUP
for p in "${!ORIG_CPU[@]}"; do w $p/scaling_governor performance; done
die() { echo "ABORT: $*" | tee -a "$OUT/campaign.log"; exit 1; }
set_npu() { w $DF/governor performance; w $DF/max_freq "$1"; sleep 1
            [ "$(cat $DF/cur_freq)" = "$1" ] || die "npu pin $1 read $(cat $DF/cur_freq)"; }
set_dmc() {  # od | pin
    if [ "$1" = pin ]; then w $DM/governor performance; sleep 1
        [ "$(cat $DM/cur_freq)" = 2112000000 ] || die "dmc pin read $(cat $DM/cur_freq)"
    else w $DM/governor dmc_ondemand; sleep 1
        [ "$(cat $DM/governor)" = dmc_ondemand ] || die "dmc od read $(cat $DM/governor)"; fi
}
reset_mem() { sync; w /proc/sys/vm/drop_caches 3; w /proc/sys/vm/compact_memory 1; sleep 2; }
sampler() { while :; do cat $DM/cur_freq; sleep 0.05; done > "$1"; }
hist() { sort "$1" | uniq -c | awk '{printf "%s@%dMHz ", $1, $2/1e6}'; }

run_arm() {  # $1 arm, $2 dmc, $3 npu MHz, $4 pass
    local log="$OUT/p$4-$2-$3-$1.log" smp="$OUT/p$4-$2-$3-$1.ddr" v
    reset_mem
    sampler "$smp" & local sp=$!
    case $1 in
        rknn3) taskset -c 4-7 $E/rknn_enc_bench $M 012 5 20 > "$log" 2>&1
               v=$(awk '/^RESULT/{print $5}' "$log");;
        ours)  GGML_BACKEND_PATH=$SO ROCKET_KACC=1 taskset -c 4-7 $WC/build/bin/whisper-cli \
                   -m $WC/models/ggml-base.en.bin -f $WC/samples/jfk.wav -ac 1000 -t 4 > "$log" 2>&1
               v=$(grep -oE 'encode time = +[0-9.]+' "$log" | awk '{print $4}')
               grep -q 'using ROCKET backend' "$log" || v="$v NO-ROCKET";;
        cpu)   taskset -c 4-7 $WC/build/bin/whisper-cli \
                   -m $WC/models/ggml-base.en.bin -f $WC/samples/jfk.wav -ac 1000 -t 4 > "$log" 2>&1
               v=$(grep -oE 'encode time = +[0-9.]+' "$log" | awk '{print $4}');;
    esac
    kill $sp 2>/dev/null; wait $sp 2>/dev/null
    printf '%s\t%s\t%s\t%s\t%s\tddr: %s\n' "$4" "$2" "$3" "$1" "${v:-FAIL}" "$(hist "$smp")" | tee -a "$OUT/results.tsv"
}

echo "[$(date -Is)] start: $PASSES passes" | tee "$OUT/campaign.log"
for ((p = 1; p <= PASSES; p++)); do
    if ((p % 2)); then DMCS=(od pin); else DMCS=(pin od); fi
    for d in "${DMCS[@]}"; do
        set_dmc "$d"
        set_npu 1000000000
        A=(rknn3 ours cpu); n=${#A[@]}
        for ((i = 0; i < n; i++)); do run_arm "${A[$(((i + p) % n))]}" "$d" 1000 "$p"; done
        set_npu 600000000
        run_arm rknn3 "$d" 600 "$p"
    done
done
echo "[$(date -Is)] CAMPAIGN-DONE" | tee -a "$OUT/campaign.log"
