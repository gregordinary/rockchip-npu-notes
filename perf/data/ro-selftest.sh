#!/bin/bash
# ro-selftest.sh -- the positive controls the per-process readout has to pass before any flat
# column it produces is read as a null. Two parts, because the readout has two independent halves:
#
#   PART 1, placement (ro-pagemap.py).  On an idle RK1 that the bench harness has just reset
#   (drop_caches + compact_memory), physical placement is essentially perfect -- pages come back
#   in runs of hundreds and the cache colours uniform -- so every placement column reads the same
#   value on every arm. A flat column there cannot be told apart from an instrument that is not
#   reading anything. The contrast has to SHATTER the buddy allocator, and filling the page cache
#   does NOT do it: clean page cache is reclaimed in whole high-order blocks, so that cell comes
#   back MORE contiguous than the compacted one (measured: mean_run 319 against 122). What works
#   is holding every other 4 KB page of a large mapping, which leaves nothing above order 0 to
#   coalesce.
#
#   PART 2, thread placement and the PMU (bench-llm.sh's <!--RO--> line).  An A55 retires roughly
#   a third of what an A76 does here, so which cluster the threads land on is the loudest thing
#   the readout could find. `taskset` forces the two extremes through the real harness, which
#   exercises the whole path -- perf, the per-CPU jiffy delta and the pagemap sampler -- on a real
#   llama-bench process rather than on a synthetic one.
#
# Leaves the board reset. About 12 minutes.
set -u
HERE=${HERE:-$(cd "$(dirname "$0")" \&\& pwd)}
MODEL=${MODEL:-$HERE/qwen35/Qwen3.5-0.8B-Q4_K_M.gguf}
REAL=${REAL:-$HOME/npu/llama.cpp/build/bin/llama-bench}
OUT=${OUT:-$HERE/ro-selftest-results.md}
reset_mem() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
              echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null; sleep 2; }
buddy() { awk '/zone   Normal/{for(i=5;i<=NF;i++) printf "%s ", $i}' /proc/buddyinfo; }

victim() {  # $1 = GiB of anon to allocate and hold
  python3 -c "
import time
a = bytearray($1*1024*1024*1024)
for i in range(0, len(a), 4096): a[i] = 1
time.sleep(600)" &
}
# Hold every OTHER 4 KB page of a large mapping. Freeing the alternate pages back one at a time
# is what shatters the free lists: nothing left can coalesce past order 0.
shatter() {  # $1 = GiB to map (holds half). Marker file, not a fixed sleep: how long the
             # fault-in and the madvise sweep take is a property of the board, and a control
             # that is sampled before it has finished shattering is not a control.
  rm -f /tmp/ro-shattered
  python3 -c "
import mmap, time
SZ = $1*1024*1024*1024
CH = 1 << 26
m = mmap.mmap(-1, SZ)
buf = b'x' * CH
for off in range(0, SZ, CH): m[off:off+CH] = buf          # fault in at memcpy speed
for off in range(4096, SZ, 8192): m.madvise(mmap.MADV_DONTNEED, off, 4096)
open('/tmp/ro-shattered','w').write('1')
time.sleep(900)" &
}
sample() {  # $1 = label -- waits for RSS to settle, samples, kills the victim
  local vp= i=0 prev=0 now=0 stable=0
  while [ $i -lt 300 ]; do
    vp=$(pgrep -x python3 | tail -1); [ -n "$vp" ] && break; i=$((i+1)); sleep 0.2
  done
  i=0
  while [ $i -lt 120 ] && [ -d "/proc/$vp" ]; do
    sleep 2; i=$((i+1)); now=$(awk '{print $2}' /proc/"$vp"/statm 2>/dev/null || echo 0)
    [ "${now:-0}" -gt 0 ] || break
    if [ "$prev" -gt 0 ] && [ $((now-prev)) -lt $((now/50)) ]; then
      stable=$((stable+1)); [ $stable -ge 2 ] && break
    else stable=0; fi
    prev=$now
  done
  echo "== $1   rss_pg=$now   buddy=$(buddy)" | tee -a "$OUT"
  sudo -n python3 "$HERE/ro-pagemap.py" "$vp" --label "$1" \
    | tr '\t' '\n' | sed 's/^/   /' | tee -a "$OUT"
}

echo "### ro readout selftest  $(date)" | tee -a "$OUT"
echo "## part 1: placement columns, compacted against shattered" | tee -a "$OUT"
reset_mem
victim 4; sample compacted
pkill -x python3; sleep 2; reset_mem
shatter 16
i=0; while [ $i -lt 600 ] && [ ! -f /tmp/ro-shattered ]; do i=$((i+1)); sleep 1; done
echo "   shatter took ${i}s; buddy after shatter: $(buddy)" | tee -a "$OUT"
victim 4; sample shattered
pkill -x python3; sleep 2; reset_mem

echo "## part 2: thread placement through the real harness (taskset extremes)" | tee -a "$OUT"
mkdir -p "$HERE/ro-wrap"
export SO=$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so
export TESTS="-p 512 -n 0 -r 2"
export PASSES=1
export ERRD="$HERE/ro-selftest-err"
export OUT
for cell in "a76only:4-7" "a55only:0-3" "free:0-7"; do
  name=${cell%%:*}; cpus=${cell##*:}
  printf '#!/bin/sh\nexec taskset -c %s %s "$@"\n' "$cpus" "$REAL" > "$HERE/ro-wrap/llama-bench"
  chmod +x "$HERE/ro-wrap/llama-bench"
  export BIN="$HERE/ro-wrap/llama-bench"
  export ARMS="$name||"
  bash "$HERE/bench-llm.sh" "$MODEL" "ro-$name"
done
reset_mem
echo "### selftest done, board reset  $(date +%T)" | tee -a "$OUT"
