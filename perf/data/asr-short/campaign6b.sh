#!/bin/bash
# Niced neighbor: the neighbor at nice 19, tg16 x 2 runs, an 800-utterance stream: does an STT offload hand CPU to another task?
# A continuous stream of b02 utterances (parakeet-cli, one process) runs beside llama-bench decode
# (Llama-3.2-3B Q4_K_M, CPU only, repack on) on the same four A76 cores. Arms rotated per pass.
set -u
W=/path/to/data/rebase/whisper.cpp-v1.9.4
L=/path/to/data/rebase/llama.cpp-b11242
M=/path/to/data/parakeet-wcpp/ggml-parakeet-tdt-0.6b-v3-f16.bin
LM=/path/to/data/llama32-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf
C=/path/to/data/stt-short/clips
O=${O:-/path/to/data/stt-short/run6b}
REPS=${REPS:-2}
ARMS=(alone cpu forced forcedp)
mkdir -p $O
FILES=$(for i in $(seq 1 80); do ls $C/b02/*.flac; done)      # 800 utterances, far longer than the neighbor
envfor() { case $1 in
  cpu)     echo "ROCKET_MIN_M=1000000 ROCKET_FLASH_ATTN=0";;
  forced)  echo "ROCKET_MIN_M=4";;
  forcedp) echo "ROCKET_MIN_M=4 OMP_WAIT_POLICY=PASSIVE";;
esac; }
arm() { # arm rep
  local a=$1 r=$2 tag=$1.r$2 sp=""
  if [ $a != alone ]; then
    env $(envfor $a) LD_LIBRARY_PATH=$W/build/bin GGML_BACKEND_PATH=/path/to/data/rebase/gr2-whisper/libggml-rocket.so \
      taskset -c 4-7 $W/build/bin/parakeet-cli -t 4 -m $M $FILES > $O/$tag.stt.out 2> $O/$tag.stt.err &
    sp=$!; sleep 4                                  # past the model load, into the stream
  fi
  local g0; g0=$(grep -c "Processing file" $O/$tag.stt.err 2>/dev/null || echo 0)
  LD_LIBRARY_PATH=$L/build/bin nice -n 19 taskset -c 4-7 $L/build/bin/llama-bench -m $LM -p 0 -n 16 -r 2 -t 4 -o jsonl > $O/$tag.nb.jsonl 2> $O/$tag.nb.err
  local rc=$? g1; g1=$(grep -c "Processing file" $O/$tag.stt.err 2>/dev/null || echo 0)
  local alive=0; [ -n "$sp" ] && kill -0 $sp 2>/dev/null && alive=1
  [ -n "$sp" ] && { kill $sp; wait $sp 2>/dev/null; }
  echo "$(date -u +%T) $tag nb_rc=$rc stt_files_during=$((g1-g0)) stt_alive=$alive tg=$(grep -o '"avg_ts": *[0-9.]*' $O/$tag.nb.jsonl | head -1)"
}
{
echo "start $(date -u) gov=$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_governor)"
arm alone w0                                                   # warm-up, discarded
for r in $(seq 1 $REPS); do
  na=${#ARMS[@]}
  for k in $(seq 0 $((na-1))); do arm ${ARMS[$(( (r + k) % na ))]} $r; done
done
echo "done $(date -u)"
} >> $O/campaign.log 2>&1
