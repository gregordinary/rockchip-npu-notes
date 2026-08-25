#!/bin/bash
# Three arms, one binary: CPU, the NPU at the shipped quantized-offload floor
# (ROCKET_MIN_M_QUANT=512, tuned for LLM prefill), and the NPU at 128. A CTC ASR
# encoder's sequence length is well under 512 for a short clip, so the shipped
# floor declines every GEMM and the NPU arm is the CPU arm with extra steps.
set -u
ROOT=${ROOT:-/path/to/asr-workdir}
T=$ROOT/transcribe.cpp
SO=$ROOT/gr-transcribe/libggml-rocket.so
OUT=${1:-$ROOT/sweep-transcribe-q.tsv}
REPS=${REPS:-2}
MODELS=${MODELS:-"SenseVoiceSmall-Q8_0 parakeet-ctc-0.6b-Q8_0"}
CLIPS=${CLIPS:-"003 010 030 060 120"}
CLIPPFX=${CLIPPFX:-aunt}
exec 9>$ROOT/npu.lock; flock 9 || exit 1
printf 'model\tclip_s\tarm\trep\twall_s\tuser_s\tsys_s\tcpu_s\trt\tmd5\n' > $OUT
run_one() {
  local m=$1 c=$2 arm=$3 rep=$4 be=cpu q=512
  case $arm in npu512) be=cpu_accel; q=512;; npu128) be=cpu_accel; q=128;; esac
  local base=$ROOT/runs-q/${CLIPPFX}_${m}_${c}_${arm}_${rep}
  local t
  t=$( { TIMEFORMAT='%R %U %S'; time GGML_BACKEND_PATH=$SO ROCKET_KACC=1 ROCKET_MIN_M_QUANT=$q \
        taskset -c 4-7 $T/build/bin/transcribe-cli -m $ROOT/models/$m.gguf \
        --threads 4 --backend $be --timestamps none \
        $ROOT/clips/${CLIPPFX}_${c}s.wav > $base.out 2> $base.err; } 2>&1 | tail -1)
  local wall user sys; read wall user sys <<< "$t"
  local cpu=$(python3 -c "print(f'{$user+$sys:.3f}')")
  local rt=$(grep -oP 'realtime:\s+\K[0-9]+' $base.out | tail -1)
  local md5=$(grep -oP '^text:\s*\K.*' $base.out | tr -d ' ' | md5sum | cut -c1-12)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$m" "$c" "$arm" "$rep" \
    "$wall" "$user" "$sys" "$cpu" "${rt:-NA}" "$md5" >> $OUT
  echo "  $m ${c}s $arm rep=$rep wall=${wall}s cpu=${cpu}s rt=${rt}x md5=$md5"
}
mkdir -p $ROOT/runs-q
for m in $MODELS; do for c in $CLIPS; do
  echo "== $m ${c}s foreign: $(fuser -v /dev/dri/renderD129 2>&1 | tr '\n' ' ')"
  for a in cpu npu512 npu128; do run_one $m $c $a 0 > /dev/null; done
  for r in $(seq 1 $REPS); do
    if [ $((r % 2)) -eq 1 ]; then for a in cpu npu512 npu128; do run_one $m $c $a $r; done
    else                          for a in npu128 npu512 cpu; do run_one $m $c $a $r; done; fi
  done
done; done
echo SWEEP_DONE
