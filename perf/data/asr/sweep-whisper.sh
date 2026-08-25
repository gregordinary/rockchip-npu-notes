#!/bin/bash
# M6 -- whisper.cpp model-and-length sweep on the vendor kernel.
#
# Two arms from ONE binary: the NPU arm offloads, the CPU arm declines every op
# at supports_op (ROCKET_MIN_M huge + ROCKET_FLASH_ATTN=0) so the process layout,
# the thread count and the scheduler are identical and only the placement moves.
#
# Reports wall AND cpu core-seconds (user+sys), because the wall speedup is not
# the CPU saving: this backend leaves packA/packB/de-tile on the host by design.
set -u
ROOT=${ROOT:-/path/to/asr-workdir}
W=$ROOT/whisper.cpp
SO=$ROOT/gr-whisper/libggml-rocket.so
OUT=${1:-$ROOT/sweep-whisper.tsv}
REPS=${REPS:-3}
MODELS=${MODELS:-"tiny.en base.en small.en"}
CLIPS=${CLIPS:-"003 010 030 060 120"}
THREADS=${THREADS:-4}
CLIPPFX=${CLIPPFX:-clip}

exec 9>$ROOT/npu.lock
flock 9 || { echo "no lock"; exit 1; }

printf 'model\tclip_s\tarm\trep\twall_s\tuser_s\tsys_s\tcpu_s\tenc_ms\tdec_ms\ttotal_ms\tmd5\n' > $OUT

run_one() {  # model clip arm rep
  local m=$1 c=$2 arm=$3 rep=$4
  local env_common="GGML_BACKEND_PATH=$SO ROCKET_KACC=1"
  local env_arm=""
  [ "$arm" = cpu ] && env_arm="ROCKET_MIN_M=1000000 ROCKET_FLASH_ATTN=0"
  local base=$ROOT/runs/${CLIPPFX}_${m}_${c}_${arm}_${rep}
  local t
  t=$( { TIMEFORMAT='%R %U %S'; time env $env_common $env_arm \
        taskset -c 4-7 $W/build/bin/whisper-cli \
          -m $W/models/ggml-$m.bin -f $ROOT/clips/${CLIPPFX}_${c}s.wav \
          -t $THREADS -nt > $base.txt 2> $base.err ; } 2>&1 | tail -1 )
  local wall user sys
  read wall user sys <<< "$t"
  local cpu=$(python3 -c "print(f'{$user+$sys:.3f}')")
  local enc dec tot
  enc=$(grep -oP 'encode time =\s*\K[0-9.]+' $base.err | tail -1)
  dec=$(grep -oP 'decode time =\s*\K[0-9.]+' $base.err | tail -1)
  tot=$(grep -oP 'total time =\s*\K[0-9.]+' $base.err | tail -1)
  local md5=$(tr -d ' \n' < $base.txt | md5sum | cut -c1-12)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$m" "$c" "$arm" "$rep" "$wall" "$user" "$sys" "$cpu" "${enc:-NA}" "${dec:-NA}" "${tot:-NA}" "$md5" >> $OUT
  echo "  $m clip=${c}s $arm rep=$rep wall=${wall}s cpu=${cpu}s enc=${enc}ms md5=$md5"
}

mkdir -p $ROOT/runs
for m in $MODELS; do
  for c in $CLIPS; do
    echo "== $m clip=${c}s  foreign: $(fuser -v /dev/dri/renderD129 2>&1 | tr '\n' ' ')"
    # warm-up, discarded: the NPU clock and the page cache both park cold
    run_one $m $c npu 0 > /dev/null
    run_one $m $c cpu 0 > /dev/null
    for r in $(seq 1 $REPS); do
      if [ $((r % 2)) -eq 1 ]; then run_one $m $c npu $r; run_one $m $c cpu $r
      else                          run_one $m $c cpu $r; run_one $m $c npu $r; fi
    done
  done
done
echo SWEEP_DONE
