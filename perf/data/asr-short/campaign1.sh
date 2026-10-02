#!/bin/bash
# Short-utterance STT campaign. parakeet-cli, Parakeet TDT 0.6B v3 F16.
# Arms x buckets x {1-file, 10-file process} x REPS, arm order rotated per (rep, bucket).
# Per-utterance cost = (10-file process - 1-file process) / 9.
set -u
W=/path/to/data/rebase/whisper.cpp-v1.9.4
export LD_LIBRARY_PATH=$W/build/bin GGML_BACKEND_PATH=/path/to/data/rebase/gr2-whisper/libggml-rocket.so
M=/path/to/data/parakeet-wcpp/ggml-parakeet-tdt-0.6b-v3-f16.bin
C=/path/to/data/stt-short/clips
O=${O:-/path/to/data/stt-short/run1}
REPS=${REPS:-3}
ARMS=(cpu cpup def forced forcedp)
BUCKETS=(b01 b02 b03 b05 b08 b12)
mkdir -p $O
envfor() { case $1 in
  cpu)     echo "ROCKET_MIN_M=1000000 ROCKET_FLASH_ATTN=0";;
  cpup)    echo "ROCKET_MIN_M=1000000 ROCKET_FLASH_ATTN=0 OMP_WAIT_POLICY=PASSIVE";;
  def)     echo "ROCKET_STT_ARM=def";;
  forced)  echo "ROCKET_MIN_M=4";;
  forcedp) echo "ROCKET_MIN_M=4 OMP_WAIT_POLICY=PASSIVE";;
esac; }
run() { # arm bucket n rep
  local arm=$1 b=$2 n=$3 r=$4; local tag=$arm.$b.n$n.r$r
  local files; files=$(ls $C/$b/*.flac | head -$n)
  env $(envfor $arm) perf stat -x, -e task-clock,armv8_cortex_a76/cycles/u,armv8_cortex_a55/cycles/u -o $O/$tag.perf -- \
    /usr/bin/time -o $O/$tag.time -f "%e %U %S" taskset -c 4-7 $W/build/bin/parakeet-cli -t 4 -m $M $files > $O/$tag.out 2> $O/$tag.err
  local rc=$?
  echo "$(date -u +%T) $tag rc=$rc $(cat $O/$tag.time) load=$(cat /proc/loadavg | cut -d' ' -f1)"
}
{
echo "start $(date -u) npu_clk=$(cat /sys/module/rocket/parameters/rocket_npu_clk_hz) gov=$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_governor) autosusp=$(cat /sys/bus/platform/drivers/rocket/fdab0000.npu/power/autosuspend_delay_ms)"
run cpu b02 1 w0; run def b02 1 w0; run forced b02 1 w0      # warm-ups, discarded
for r in $(seq 1 $REPS); do
  bi=0
  for b in "${BUCKETS[@]}"; do
    na=${#ARMS[@]}; off=$(( (r + bi) % na ))
    for k in $(seq 0 $((na-1))); do
      arm=${ARMS[$(( (off + k) % na ))]}
      if (( r % 2 )); then run $arm $b 1 $r; run $arm $b 10 $r; else run $arm $b 10 $r; run $arm $b 1 $r; fi
    done
    bi=$((bi+1))
  done
done
echo "done $(date -u)"
} >> $O/campaign.log 2>&1
