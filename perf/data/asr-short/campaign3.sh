#!/bin/bash
# Admission map + profile pass (not timed for the ledger): which GEMMs each floor admits per bucket,
# and the ROCKET_MM_PROFILE buckets per 10-utterance process at floors 4 and 16.
set -u
W=/path/to/data/rebase/whisper.cpp-v1.9.4
export LD_LIBRARY_PATH=$W/build/bin GGML_BACKEND_PATH=/path/to/data/rebase/gr2-whisper/libggml-rocket.so
M=/path/to/data/parakeet-wcpp/ggml-parakeet-tdt-0.6b-v3-f16.bin
C=/path/to/data/stt-short/clips
O=${O:-/path/to/data/stt-short/run3}
mkdir -p $O
{
for b in b01 b02 b03 b05 b08 b12; do
  for fl in 4 8 16 32 64 128; do
    for f in $(ls $C/$b/*.flac | head -1) $(ls $C/$b/*.flac | tail -1); do
      u=$(basename $f .flac)
      ROCKET_MIN_M=$fl ROCKET_MM_PROFILE=1 taskset -c 4-7 $W/build/bin/parakeet-cli -t 4 -m $M $f > /dev/null 2> $O/adm.$b.$fl.$u.err
      echo "adm $b floor=$fl $u $(grep -o 'pack_act=[0-9]* ([0-9]* calls' $O/adm.$b.$fl.$u.err | grep -o '[0-9]* calls' || echo '0 calls')"
    done
  done
done
for b in b01 b02 b05 b12; do
  for fl in 4 16; do
    ROCKET_MIN_M=$fl ROCKET_MM_PROFILE=1 taskset -c 4-7 $W/build/bin/parakeet-cli -t 4 -m $M $(ls $C/$b/*.flac) > $O/prof.$b.$fl.out 2> $O/prof.$b.$fl.err
    echo "prof $b floor=$fl $(grep 'ROCKET profile' $O/prof.$b.$fl.err)"
  done
done
echo "done $(date -u)"
} >> $O/campaign.log 2>&1
