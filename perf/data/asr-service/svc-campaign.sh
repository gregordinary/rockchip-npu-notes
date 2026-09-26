#!/bin/bash
# The whisper-server service sweep: every arm is one server process driven by the same
# 20 s (or 30 s) radio-channel chunk sets, rotated across passes, memory reset per arm.
#
#   svc-campaign.sh OUT.tsv [PASSES] [STREAM]
set -u
ROOT=${ROOT:-/path/to/asr-workdir}
OUT=${1:-$ROOT/results/campaign.tsv}
PASSES=${2:-2}
STREAM=${3:-s61-70968}
SO_BASE=$ROOT/so/libggml-rocket-base.so
SO_FA=$ROOT/so/libggml-rocket-fa.so
M=$ROOT/models
SETS20="$STREAM/clean/c20 $STREAM/ssb15/c20 $STREAM/ssb5/c20"
SETS30="$STREAM/clean/c30 $STREAM/ssb15/c30 $STREAM/ssb5/c30"
exec 9>"${NPU_LOCK:-/tmp/npu.lock}"; flock 9 || { echo "no lock"; exit 1; }
export OUT

# arm name -> "ENV... -- sets -- server flags"
run_arm() {  # name rep
  local a=$1 r=$2
  case $a in
    cpu)              NPU=0 SO=$SO_BASE            ./svc-arm.sh $a $r "$SETS20" -- ;;
    npu)              SO=$SO_BASE                  ./svc-arm.sh $a $r "$SETS20" -- ;;
    npu-ac1000)       SO=$SO_BASE                  ./svc-arm.sh $a $r "$SETS20" -- -ac 1000 ;;
    npu-c30)          SO=$SO_BASE                  ./svc-arm.sh $a $r "$SETS30" -- ;;
    npu-fa)           SO=$SO_FA                    ./svc-arm.sh $a $r "$SETS20" -- ;;
    npu-q8)           SO=$SO_BASE MODEL=$M/ggml-small-q8_0.bin ./svc-arm.sh $a $r "$SETS20" -- ;;
    npu-q5)           SO=$SO_BASE MODEL=$M/ggml-small-q5_1.bin ./svc-arm.sh $a $r "$SETS20" -- ;;
    npu-t2)           SO=$SO_BASE THREADS=2        ./svc-arm.sh $a $r "$SETS20" -- ;;
    npu-nt)           SO=$SO_BASE                  ./svc-arm.sh $a $r "$SETS20" -- -nt ;;
    npu-nf)           SO=$SO_BASE                  ./svc-arm.sh $a $r "$SETS20" -- -nf ;;
    npu-unpinned)     SO=$SO_BASE CPUS=""          ./svc-arm.sh $a $r "$SETS20" -- ;;
    *) echo "unknown arm $a"; return 1 ;;
  esac
}
ARMS=${ARMS:-"cpu npu npu-ac1000 npu-c30 npu-fa npu-q8 npu-q5 npu-t2 npu-nt npu-nf npu-unpinned"}
cd $ROOT
for p in $(seq 1 $PASSES); do
  order="$ARMS"
  [ $((p % 2)) -eq 0 ] && order=$(echo $ARMS | tr ' ' '\n' | tac | tr '\n' ' ')
  for a in $order; do
    echo "== pass $p arm $a  $(date +%T)  foreign: $(sudo -n fuser -v /dev/accel/accel0 2>&1 | tr '\n' ' ')"
    run_arm $a $p 2>&1 | grep -E "^arm=|ARM_DONE|did not come up"
  done
done
echo CAMPAIGN_DONE
