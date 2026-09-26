#!/bin/bash
# Follow-up service sweep: the real no-fallback arm (per-request temperature_inc=0, the
# field OpenWebRX+ can send; whisper-server's -nf flag is parsed and never applied) and the
# stacked candidate configurations, against the shipped NPU arm. Same shape as svc-campaign.sh.
#
#   svc-campaign2.sh OUT.tsv [PASSES] [STREAM]      ARMS="..." selects the arms
set -u
ROOT=${ROOT:-/path/to/asr-workdir}
OUT=${1:-$ROOT/results/campaign2.tsv}
PASSES=${2:-2}
STREAM=${3:-s61-70968}
SO_BASE=$ROOT/so/libggml-rocket-base.so
M=$ROOT/models
SETS20="$STREAM/clean/c20 $STREAM/ssb15/c20 $STREAM/ssb5/c20"
SETS30="$STREAM/clean/c30 $STREAM/ssb15/c30 $STREAM/ssb5/c30"
SETS25="$STREAM/clean/c25 $STREAM/ssb15/c25 $STREAM/ssb5/c25"
exec 9>"${NPU_LOCK:-/tmp/npu.lock}"; flock 9 || { echo "no lock"; exit 1; }
export OUT
NF="temperature_inc=0"

run_arm() {  # name rep
  local a=$1 r=$2
  case $a in
    cpu)               NPU=0 SO=$SO_BASE                 ./svc-arm.sh $a $r "$SETS20" -- ;;
    npu)               SO=$SO_BASE                       ./svc-arm.sh $a $r "$SETS20" -- ;;
    npu-nf)            SO=$SO_BASE CLIENT_FIELDS=$NF     ./svc-arm.sh $a $r "$SETS20" -- ;;
    npu-ac1000-nf)     SO=$SO_BASE CLIENT_FIELDS=$NF     ./svc-arm.sh $a $r "$SETS20" -- -ac 1000 ;;
    npu-q8-nf)         SO=$SO_BASE CLIENT_FIELDS=$NF MODEL=$M/ggml-small-q8_0.bin ./svc-arm.sh $a $r "$SETS20" -- ;;
    npu-q8-ac1000-nf)  SO=$SO_BASE CLIENT_FIELDS=$NF MODEL=$M/ggml-small-q8_0.bin ./svc-arm.sh $a $r "$SETS20" -- -ac 1000 ;;
    npu-ac1000-nf-nt)  SO=$SO_BASE CLIENT_FIELDS=$NF     ./svc-arm.sh $a $r "$SETS20" -- -ac 1000 -nt ;;
    npu-ac1000-nt)     SO=$SO_BASE                       ./svc-arm.sh $a $r "$SETS20" -- -ac 1000 -nt ;;
    npu-q8-ac1000-nf-nt) SO=$SO_BASE CLIENT_FIELDS=$NF MODEL=$M/ggml-small-q8_0.bin ./svc-arm.sh $a $r "$SETS20" -- -ac 1000 -nt ;;
    npu-q5-ac1000-nf-nt) SO=$SO_BASE CLIENT_FIELDS=$NF MODEL=$M/ggml-small-q5_1.bin ./svc-arm.sh $a $r "$SETS20" -- -ac 1000 -nt ;;
    npu-ac1000-nf-nt-t2) SO=$SO_BASE CLIENT_FIELDS=$NF THREADS=2 ./svc-arm.sh $a $r "$SETS20" -- -ac 1000 -nt ;;
    npu-ac1000-nf-nt-sns) SO=$SO_BASE CLIENT_FIELDS=$NF    ./svc-arm.sh $a $r "$SETS20" -- -ac 1000 -nt -sns ;;
    npu-ac1250-c25-nf-nt) SO=$SO_BASE CLIENT_FIELDS=$NF    ./svc-arm.sh $a $r "$SETS25" -- -ac 1250 -nt ;;
    npu-ac1250-c25-nf-nt-sns) SO=$SO_BASE CLIENT_FIELDS=$NF ./svc-arm.sh $a $r "$SETS25" -- -ac 1250 -nt -sns ;;
    npu-c30-nf-nt-sns) SO=$SO_BASE CLIENT_FIELDS=$NF     ./svc-arm.sh $a $r "$SETS30" -- -nt -sns ;;
    npu-ac1200-nf-nt)  SO=$SO_BASE CLIENT_FIELDS=$NF     ./svc-arm.sh $a $r "$SETS20" -- -ac 1200 -nt ;;
    npu-ac1200-nf)     SO=$SO_BASE CLIENT_FIELDS=$NF     ./svc-arm.sh $a $r "$SETS20" -- -ac 1200 ;;
    npu-ac1200-nf-nt-sns) SO=$SO_BASE CLIENT_FIELDS=$NF  ./svc-arm.sh $a $r "$SETS20" -- -ac 1200 -nt -sns ;;
    npu-ac1250-nf-nt)  SO=$SO_BASE CLIENT_FIELDS=$NF     ./svc-arm.sh $a $r "$SETS20" -- -ac 1250 -nt ;;
    npu-c25-nf-nt)     SO=$SO_BASE CLIENT_FIELDS=$NF     ./svc-arm.sh $a $r "$SETS25" -- -nt ;;
    npu-nf-nt)         SO=$SO_BASE CLIENT_FIELDS=$NF     ./svc-arm.sh $a $r "$SETS20" -- -nt ;;
    npu-q8-c30-nf)     SO=$SO_BASE CLIENT_FIELDS=$NF MODEL=$M/ggml-small-q8_0.bin ./svc-arm.sh $a $r "$SETS30" -- ;;
    npu-c30-nf)        SO=$SO_BASE CLIENT_FIELDS=$NF     ./svc-arm.sh $a $r "$SETS30" -- ;;
    npu-c25)           SO=$SO_BASE                       ./svc-arm.sh $a $r "$SETS25" -- ;;
    npu-c25-nf)        SO=$SO_BASE CLIENT_FIELDS=$NF     ./svc-arm.sh $a $r "$SETS25" -- ;;
    npu-ac1250-c25-nf) SO=$SO_BASE CLIENT_FIELDS=$NF     ./svc-arm.sh $a $r "$SETS25" -- -ac 1250 ;;
    npu-q8-ac1250-c25-nf) SO=$SO_BASE CLIENT_FIELDS=$NF MODEL=$M/ggml-small-q8_0.bin ./svc-arm.sh $a $r "$SETS25" -- -ac 1250 ;;
    npu-c30-nf-nt)     SO=$SO_BASE CLIENT_FIELDS=$NF     ./svc-arm.sh $a $r "$SETS30" -- -nt ;;
    npu-c30-nt)        SO=$SO_BASE                       ./svc-arm.sh $a $r "$SETS30" -- -nt ;;
    npu-ac1000-t2-nf)  SO=$SO_BASE CLIENT_FIELDS=$NF THREADS=2 ./svc-arm.sh $a $r "$SETS20" -- -ac 1000 ;;
    *) echo "unknown arm $a"; return 1 ;;
  esac
}
ARMS=${ARMS:-"npu npu-nf npu-c25 npu-c25-nf npu-ac1000-nf npu-ac1000-nf-nt npu-ac1000-nf-nt-t2 npu-q5-ac1000-nf-nt npu-ac1250-c25-nf npu-c30-nf-nt"}
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
