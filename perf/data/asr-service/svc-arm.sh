#!/bin/bash
# One arm of the whisper-server service benchmark: start a server with this arm's
# environment and flags, drive it with svc-client.py over every requested set, stop it.
#
#   svc-arm.sh ARM REP "SET1 SET2 ..." -- [whisper-server flags]
#
# Environment knobs (all optional):
#   MODEL     model file                          (default ggml-small.bin)
#   THREADS   whisper -t                          (default 4)
#   CPUS      taskset list for the server, "" = unpinned   (default 4-7)
#   NPU       1 = offload (default), 0 = CPU arm from the same binary
#   SO        libggml-rocket.so path
#   ARMENV    extra "K=V K=V" for the server process (ROCKET_* knobs)
#   WARM      warm-up requests to discard per set (default 2 on the first set only)
#   PORT      (default 8090)
set -u
ROOT=${ROOT:-/path/to/asr-workdir}
W=$ROOT/src/whisper.cpp
SO=${SO:-$ROOT/src/ggml-rocket/build-whisper/libggml-rocket.so}
MODEL=${MODEL:-$ROOT/models/ggml-small.bin}
THREADS=${THREADS:-4}
CPUS=${CPUS-4-7}
NPU=${NPU:-1}
ARMENV=${ARMENV:-}
PORT=${PORT:-8090}
WARM=${WARM:-2}
ARM=$1; REP=$2; SETS=$3; shift 3
[ "${1:-}" = "--" ] && shift
OUT=${OUT:-$ROOT/results/results.tsv}
mkdir -p $(dirname $OUT) $ROOT/results/logs
PY=${PY:-python3}

env_arm="GGML_BACKEND_PATH=$SO ROCKET_KACC=1"
[ "$NPU" = 0 ] && env_arm="$env_arm ROCKET_MIN_M=1000000 ROCKET_FLASH_ATTN=0"
pin=""; [ -n "$CPUS" ] && pin="taskset -c $CPUS"

# memory reset before the arm (the board's allocation history moves timings by ~8%)
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'

log=$ROOT/results/logs/${ARM}_${REP}.server.log
sudo -n env $env_arm $ARMENV $pin $W/build/bin/whisper-server -m $MODEL -t $THREADS \
    --host 127.0.0.1 --port $PORT "$@" > $log 2>&1 &
sudo_pid=$!
# the server PID is the whisper-server child of sudo
for i in $(seq 1 600); do
    spid=$(pgrep -P $sudo_pid whisper-server 2>/dev/null | head -1)
    [ -z "$spid" ] && spid=$(pgrep -f "whisper-server -m $MODEL" | head -1)
    if [ -n "$spid" ] && curl -s -o /dev/null -m 2 http://127.0.0.1:$PORT/ ; then break; fi
    sleep 0.5
done
if [ -z "$spid" ] || ! curl -s -o /dev/null -m 2 http://127.0.0.1:$PORT/ ; then
    echo "server did not come up (arm $ARM)"; tail -20 $log; sudo -n kill $sudo_pid 2>/dev/null; exit 1
fi
echo "arm=$ARM rep=$REP pid=$spid env='$env_arm $ARMENV' pin='$pin' flags='$*'"
grep -E "ROCKET|device|system_info|DOTPROD" $log | head -5

first=1
for s in $SETS; do
    w=0; [ $first = 1 ] && w=$WARM; first=0
    $PY $ROOT/svc-client.py --url http://127.0.0.1:$PORT/inference --pid $spid \
        --set $s --dir $ROOT/audio/$s --out $OUT --arm $ARM --rep $REP --warm $w ${CLIENT_FIELDS:+--field $CLIENT_FIELDS}
done
sudo -n kill $spid; wait $sudo_pid 2>/dev/null
sleep 1
echo "ARM_DONE $ARM $REP"
