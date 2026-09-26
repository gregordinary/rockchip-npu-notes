#!/bin/bash
# svc-arm.sh adapted to the vendor RK1: this board's paths, the server run as the invoking user
# (render + video cover the NPU), and the memory reset through the password sudo helper.
#   PW=... svc-arm-vendor.sh ARM REP "SET1 SET2 ..." -- [whisper-server flags]
set -u
ROOT=${ROOT:-/path/to/asr-workdir}
F=${F:?set F to the directory holding whisper.cpp and build-ggml-rocket}
W=$F/whisper.cpp
SO=${SO:-$F/build-ggml-rocket/libggml-rocket.so}
MODEL=${MODEL:-$ROOT/models/ggml-small.bin}
THREADS=${THREADS:-4}; CPUS=${CPUS-4-7}; NPU=${NPU:-1}; ARMENV=${ARMENV:-}; PORT=${PORT:-8090}; WARM=${WARM:-2}
ARM=$1; REP=$2; SETS=$3; shift 3
[ "${1:-}" = "--" ] && shift
OUT=${OUT:-$ROOT/results/results.tsv}
mkdir -p $(dirname $OUT) $ROOT/results/logs
env_arm="GGML_BACKEND_PATH=$SO ROCKET_KACC=1"
[ "$NPU" = 0 ] && env_arm="$env_arm ROCKET_MIN_M=1000000 ROCKET_FLASH_ATTN=0"
pin=""; [ -n "$CPUS" ] && pin="taskset -c $CPUS"
printf '%s\n' "$PW" | sudo -S -p '' sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'
log=$ROOT/results/logs/${ARM}_${REP}.server.log
env $env_arm $ARMENV $pin $W/build/bin/whisper-server -m $MODEL -t $THREADS \
    --host 127.0.0.1 --port $PORT "$@" > $log 2>&1 &
spid=$!
for i in $(seq 1 600); do curl -s -o /dev/null -m 2 http://127.0.0.1:$PORT/ && break; sleep 0.5; done
if ! curl -s -o /dev/null -m 2 http://127.0.0.1:$PORT/ ; then
    echo "server did not come up (arm $ARM)"; tail -20 $log; kill $spid 2>/dev/null; exit 1
fi
echo "arm=$ARM rep=$REP pid=$spid env='$env_arm $ARMENV' pin='$pin' flags='$*'"
first=1
for s in $SETS; do
    w=0; [ $first = 1 ] && w=$WARM; first=0
    python3 $ROOT/svc-client.py --url http://127.0.0.1:$PORT/inference --pid $spid \
        --set $s --dir $ROOT/audio/$s --out $OUT --arm $ARM --rep $REP --warm $w ${CLIENT_FIELDS:+--field $CLIENT_FIELDS}
done
kill $spid; wait $spid 2>/dev/null; sleep 1
echo "ARM_DONE $ARM $REP"
