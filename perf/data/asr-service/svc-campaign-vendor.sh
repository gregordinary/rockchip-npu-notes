#!/bin/bash
# The two recommended service configurations on the vendor RK1, against the shipped NPU arm and
# the CPU arm, stream s61-70968, arm order reversed on even passes. CPU governor performance,
# NPU 1000 MHz and DDR 2112 MHz pinned and read back, all restored on exit.
#   PW=... svc-campaign-vendor.sh OUT.tsv [PASSES]
set -u
ROOT=${ROOT:-/path/to/asr-workdir}
export OUT=${1:-$ROOT/results/vendor.tsv}; PASSES=${2:-2}; STREAM=s61-70968
SETS20="$STREAM/clean/c20 $STREAM/ssb15/c20 $STREAM/ssb5/c20"
SETS30="$STREAM/clean/c30 $STREAM/ssb15/c30 $STREAM/ssb5/c30"
NF="temperature_inc=0"
DF=/sys/class/devfreq/fdab0000.npu; DM=/sys/class/devfreq/dmc
S() { printf '%s\n' "$PW" | sudo -S -p '' "$@"; }
w() { S sh -c 'printf "%s" "$2" > "$1"' _ "$1" "$2"; }
ORIG_NPU_GOV=$(cat $DF/governor); ORIG_NPU_MAX=$(cat $DF/max_freq); ORIG_DMC_GOV=$(cat $DM/governor)
declare -A ORIG_CPU
for p in /sys/devices/system/cpu/cpufreq/policy*; do ORIG_CPU[$p]=$(cat $p/scaling_governor); done
restore() { w $DF/max_freq "$ORIG_NPU_MAX"; w $DF/governor "$ORIG_NPU_GOV"; w $DM/governor "$ORIG_DMC_GOV"
            for p in "${!ORIG_CPU[@]}"; do w $p/scaling_governor "${ORIG_CPU[$p]}"; done
            echo "[$(date -Is)] restored: npu $(cat $DF/governor) dmc $(cat $DM/governor) cpu $(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_governor)"; }
trap restore EXIT; trap 'exit 1' INT TERM HUP
for p in "${!ORIG_CPU[@]}"; do w $p/scaling_governor performance; [ "$(cat $p/scaling_governor)" = performance ] || { echo ABORT cpu gov; exit 1; }; done
w $DF/governor performance; w $DF/max_freq 1000000000; w $DM/governor performance; sleep 1
[ "$(cat $DF/cur_freq)" = 1000000000 ] && [ "$(cat $DM/cur_freq)" = 2112000000 ] || { echo "ABORT pins: npu $(cat $DF/cur_freq) dmc $(cat $DM/cur_freq)"; exit 1; }
cd $ROOT
run_arm() {
  local a=$1 r=$2
  case $a in
    cpu)                  NPU=0                         ./svc-arm-vendor.sh $a $r "$SETS20" -- ;;
    npu)                                                ./svc-arm-vendor.sh $a $r "$SETS20" -- ;;
    npu-ac1200-nf-nt-sns) CLIENT_FIELDS=$NF             ./svc-arm-vendor.sh $a $r "$SETS20" -- -ac 1200 -nt -sns ;;
    npu-c30-nf-nt)        CLIENT_FIELDS=$NF             ./svc-arm-vendor.sh $a $r "$SETS30" -- -nt ;;
  esac
}
ARMS="npu cpu npu-ac1200-nf-nt-sns npu-c30-nf-nt"
for p in $(seq 1 $PASSES); do
  order="$ARMS"; [ $((p % 2)) -eq 0 ] && order=$(echo $ARMS | tr ' ' '\n' | tac | tr '\n' ' ')
  for a in $order; do
    echo "== pass $p arm $a $(date +%T) npu=$(cat $DF/cur_freq) dmc=$(cat $DM/cur_freq) gov=$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_governor)"
    run_arm $a $p 2>&1 | grep -E "^arm=|ARM_DONE|did not come up"
  done
done
echo CAMPAIGN_DONE
