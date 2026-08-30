#!/bin/bash
# moe-ballast-ladder.sh -- INDUCE the MoE budget regime on a second model, and read the TURN.
#
# The MoE budget hypothesis is that the per-expert admission charge omits the shared expert
# GGUF, so the route over-places once the true hot set approaches RAM. It has an arithmetic fit
# to five rungs on ONE model, and no second model on this board reaches the regime unaided:
# gpt-oss-20b's hot set at full placement is ~14 GB anonymous + 11.27 GiB GGUF = ~25.3 GB
# against 31.7 GB of RAM. So add ballast B and walk it up.
#
# THE PREDICTED QUANTITY IS `B`, NOT A RATIO. 25.3 + B > 31.7 puts the turn at B ~ 6-7 GB, so a
# ladder at 0 / 4 / 8 confirms the account if it turns between 4 and 8, and refutes it if it
# turns at 4 or does not turn at 8. Read the TURN: the predicted collapse is into the measured
# 0.42-0.97x partial-residency regime, far outside this arm's spread, while the rung-to-rung
# percentages are NOT -- gpt-oss's own pp512 column carries a +/-1.4-2.1 t/s within-process
# spread. A ladder cannot report a few percent here and does not have to.
#
# THE RIVAL IS ON RECORD: ballast is anonymous and unreclaimable where the GGUF's pages are
# file-backed, so a turn at the predicted point is also consistent with "any memory pressure at
# all" rather than with the charge omission. Separating them needs a FILE-BACKED ballast of the
# same size, which would turn at a different point. Not run here.
#
# It also carries the safety item beside it: the MoE route freezes MemAvailable at the first
# supports_op and has no floor underneath it, so this is the rig on which a stale budget would
# authorise more than the board can honour. dmesg is captured per rung for that reason.
#
# Usage: MODELS=/path/to/models bash moe-ballast-ladder.sh [rungs in GiB...]   (default 0 4 8)
set -u
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
BIN=${BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
MODEL=${MODEL:-$MODELS/gpt-oss-20b/gpt-oss-20b-mxfp4.gguf}
OUTD=${OUTD:-$MODELS/moe-ballast}
BALLAST=${BALLAST:-$OUTD/ballast}
TESTS=${TESTS:--p 2048 -n 0 -r 1 -b 2048 -ub 2048}
RUNGS=${*:-0 4 8}
mkdir -p "$OUTD"

if [ ! -x "$BALLAST" ]; then
  cat > "$OUTD/ballast.c" <<'EOF'
/* ballast: hold N GiB of touched anonymous memory until killed. No swap on these boards, so
 * it is unreclaimable -- which is the point, and also the rival hypothesis (see the header). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
int main(int argc, char **argv) {
    size_t gib = argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : 0;
    if (!gib) { printf("ballast 0 GiB\n"); fflush(stdout); pause(); return 0; }
    size_t n = gib << 30;
    unsigned char *p = mmap(NULL, n, PROT_READ|PROT_WRITE,
                            MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }
    memset(p, 0xa5, n);
    printf("ballast %zu GiB resident\n", gib); fflush(stdout);
    pause();
    return 0;
}
EOF
  gcc -O2 -o "$BALLAST" "$OUTD/ballast.c" || { echo "ballast build failed"; exit 1; }
fi

echo "model $(basename "$MODEL")   tests '$TESTS'   rungs: $RUNGS"
echo
for B in $RUNGS; do
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null
  sleep 3
  "$BALLAST" "$B" > "$OUTD/ballast.$B.log" 2>&1 &
  bpid=$!
  # wait for the ballast to be RESIDENT, not merely started
  for _ in $(seq 1 200); do grep -q resident "$OUTD/ballast.$B.log" 2>/dev/null && break; sleep 1; done
  [ "$B" = 0 ] || grep -q resident "$OUTD/ballast.$B.log" || { echo "B=$B ballast never became resident"; kill $bpid; continue; }
  avail=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
  sudo dmesg -C >/dev/null 2>&1
  t0=$(date +%s)
  # shellcheck disable=SC2086
  sudo -E env GGML_BACKEND_PATH="$SO" ROCKET_KACC=1 ROCKET_LOG_STDERR=1 \
      "$BIN" -m "$MODEL" $TESTS > "$OUTD/run.$B.out" 2> "$OUTD/run.$B.err"
  t1=$(date +%s)
  sudo dmesg > "$OUTD/dmesg.$B.txt" 2>/dev/null
  kill $bpid 2>/dev/null; wait $bpid 2>/dev/null
  tps=$(awk -F'|' '/pp2048/{print $(NF-1)}' "$OUTD/run.$B.out" | tr -d ' ')
  place=$(grep -h "experts exercised\|residency pre-flight\|resident budget reached\|IOVA window full\|only .* resident" "$OUTD/run.$B.err" | tr '\n' '@' | sed 's/@/\n            /g')
  oom=$(grep -ci "out of memory\|oom-kill" "$OUTD/dmesg.$B.txt" 2>/dev/null || echo 0)
  printf 'B=%-2s GiB  MemAvailable %6.1f GB  pp2048 %-14s  wall %3ss  oom-lines %s\n' \
         "$B" "$(echo "$avail/1048576" | bc -l)" "$tps" "$((t1-t0))" "$oom"
  echo "            $place"
done
echo
echo "per-rung stderr, stdout and dmesg under $OUTD"
