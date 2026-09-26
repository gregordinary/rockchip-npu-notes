#!/bin/bash
# Validate ROCKET_FA_PROFILE before believing it: an instrument that perturbs its own subject
# is the failure mode. Both FA paths (chained and per-head), knob ON and OFF, the knob order
# rotated between passes so a board drift cannot land on one arm. Detached: every result is
# written here, not to the launching ssh's stdout.
set -u
B=$HOME/npu/rocket-userspace/build/flash_attn_rocket
OUT=/tmp/fa-prof-v2
rm -rf $OUT; mkdir -p $OUT
R=$OUT/results.txt
: > $R
run() {   # run <pass> <chain> <prof>
  local pass=$1 chain=$2 prof=$3 tag t0 t1 rc
  tag="p${pass}_chain${chain}_prof${prof}"
  local e="ROCKET_FA_CHAIN=$chain"
  [ "$prof" = 1 ] && e="$e ROCKET_FA_PROFILE=1"
  t0=$(date +%s.%N)
  sudo -E env $e $B > $OUT/$tag.log 2>&1
  rc=$?
  t1=$(date +%s.%N)
  echo "WALL $tag rc=$rc chain=$chain prof=$prof wall=$(echo "$t1-$t0" | bc)" >> $R
}
for pass in 1 2 3; do
  if [ $((pass % 2)) = 1 ]; then order="0 1"; else order="1 0"; fi
  for prof in $order; do
    for chain in 1 0; do run $pass $chain $prof; done
  done
done
{ echo "== correctness =="
  for f in $OUT/p*_prof*.log; do
    echo "--- $(basename $f) : $(grep -icE '\bFAIL' $f) FAIL lines, $(grep -icE '\bPASS|\bOK\b' $f) PASS lines"
    tail -2 $f
  done
  echo "== profile lines (prof=1 arms) =="
  for f in $OUT/p*_prof1.log; do echo "--- $(basename $f)"; grep "FA profile" $f; done
} >> $R
echo "DONE" >> $R
