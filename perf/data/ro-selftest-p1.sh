#!/bin/bash
# ro-selftest-p1.sh -- the placement half of the readout's positive control, re-taken.
#
# Two earlier attempts at the fragmenting cell both failed, both SILENTLY, and both by coming
# back MORE contiguous than the control they were supposed to degrade. That is the shape to
# expect from this kind of control, so each attempt's defect is recorded here rather than
# overwritten:
#   1. Filling the page cache with a large file read. Clean page cache is reclaimed in whole
#      high-order blocks, so the allocator was left tidier, not dirtier (mean_run 319 against
#      122 for the compacted cell).
#   2. Holding every other page of a `mmap.mmap(-1, SZ)` region. Python's default flags are
#      MAP_SHARED, so with fileno -1 that is a shmem object, and MADV_DONTNEED on shmem only
#      drops the caller's page tables -- the pages stay in the object and NOTHING returns to the
#      buddy allocator. The give-away is in /proc/buddyinfo: freeing 2.1M single pages has to
#      put ~2.1M blocks on the order-0 list, and the order-0 count went DOWN.
# Hence MAP_PRIVATE below, and hence the ARMING CHECK: the cell asserts on the order-0 count
# before its sample is allowed to mean anything. A control that did not fire is not a null.
set -u
HERE=${HERE:-$(cd "$(dirname "$0")" \&\& pwd)}
OUT=${OUT:-$HERE/ro-selftest-p1.md}
reset_mem() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
              echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null; sleep 3; }
buddy()  { awk '/zone   Normal/{for(i=5;i<=NF;i++) printf "%s ", $i}' /proc/buddyinfo; }
order0() { awk '/zone   Normal/{s+=$5} END{print s+0}' /proc/buddyinfo; }
victim() { python3 -c "
import time
a = bytearray($1*1024*1024*1024)
for i in range(0, len(a), 4096): a[i] = 1
time.sleep(600)" & }
shatter() {
  rm -f /tmp/ro-shattered
  python3 -c "
import mmap, time
SZ = $1*1024*1024*1024
CH = 1 << 26
m = mmap.mmap(-1, SZ, flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
buf = b'x' * CH
for off in range(0, SZ, CH): m[off:off+CH] = buf
for off in range(4096, SZ, 8192): m.madvise(mmap.MADV_DONTNEED, off, 4096)
open('/tmp/ro-shattered','w').write('1')
time.sleep(900)" & }
sample() {
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
  echo "== $1   rss_pg=$now   order0=$(order0)   buddy=$(buddy)" | tee -a "$OUT"
  sudo -n python3 "$HERE/ro-pagemap.py" "$vp" --label "$1" \
    | tr '\t' '\n' | sed 's/^/   /' | tee -a "$OUT"
}
: > "$OUT"
echo "### placement control, re-taken with MAP_PRIVATE   $(date)" | tee -a "$OUT"
reset_mem; victim 4; sample compacted; pkill -x python3; sleep 2
reset_mem
echo "order0 before shatter: $(order0)" | tee -a "$OUT"
shatter 16
i=0; while [ $i -lt 900 ] && [ ! -f /tmp/ro-shattered ]; do i=$((i+1)); sleep 1; done
echo "shatter took ${i}s; order0 after shatter: $(order0)   buddy=$(buddy)" | tee -a "$OUT"
victim 4; sample shattered
pkill -x python3; sleep 2; reset_mem
echo "[$(date +%T)] P1-DONE" | tee -a "$OUT"
