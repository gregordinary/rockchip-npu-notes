#!/bin/bash
# M7 cap: what fraction of an encode's HOST cost is the weight scatter?
# A streaming ASR host re-encodes against the same weights every window, and
# rocket_weight_key() rejects whisper's ggml autonames, so every weight is
# re-packed per call. mm_pack_weights() covers prep + scatter + fini, so packB
# IS the whole per-call weight cost and bounds what a weight cache can return.
set -u
ROOT=${ROOT:-/path/to/asr-workdir}
exec 9>$ROOT/npu.lock; flock 9
for m in tiny.en base.en small.en; do
  for c in 030 120; do
    # warm-up, discarded
    GGML_BACKEND_PATH=$ROOT/gr-whisper/libggml-rocket.so ROCKET_KACC=1 \
      taskset -c 4-7 $ROOT/whisper.cpp/build/bin/whisper-cli \
      -m $ROOT/whisper.cpp/models/ggml-$m.bin -f $ROOT/clips/clip_${c}s.wav -t 4 -nt >/dev/null 2>&1
    out=$( { TIMEFORMAT='%R %U %S'; time GGML_BACKEND_PATH=$ROOT/gr-whisper/libggml-rocket.so \
      ROCKET_KACC=1 ROCKET_MM_PROFILE=1 \
      taskset -c 4-7 $ROOT/whisper.cpp/build/bin/whisper-cli \
      -m $ROOT/whisper.cpp/models/ggml-$m.bin -f $ROOT/clips/clip_${c}s.wav -t 4 -nt \
      > /dev/null 2> $ROOT/prof_${m}_${c}.err ; } 2>&1 | tail -1 )
    echo "== $m clip=${c}s  wall/user/sys = $out"
    grep -E "ROCKET profile total|ROCKET convert total|encode time" $ROOT/prof_${m}_${c}.err
  done
done
echo CAP_DONE
